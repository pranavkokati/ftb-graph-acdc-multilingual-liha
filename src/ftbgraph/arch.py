import gc
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.pytorch_utils import Conv1D as HFConv1D

from . import config

ARCH = {
    "gpt2":   {"attn": "transformer.h.{L}.attn",          "block": "transformer.h.{L}",   "proj": "c_proj", "nl": "n_layer",          "nh": "n_head",            "final_norm": "transformer.ln_f"},
    "bloom":  {"attn": "transformer.h.{L}.self_attention", "block": "transformer.h.{L}",   "proj": "dense",  "nl": "n_layer",          "nh": "num_attention_heads", "final_norm": "transformer.ln_f"},
    "qwen":   {"attn": "model.layers.{L}.self_attn",       "block": "model.layers.{L}",    "proj": "o_proj", "nl": "num_hidden_layers", "nh": "num_attention_heads", "final_norm": "model.norm"},
    "pythia": {"attn": "gpt_neox.layers.{L}.attention",    "block": "gpt_neox.layers.{L}", "proj": "dense",  "nl": "num_hidden_layers", "nh": "num_attention_heads", "final_norm": "gpt_neox.final_layer_norm"},
    "aya":    {"attn": "model.layers.{L}.self_attn",       "block": "model.layers.{L}",    "proj": "o_proj", "nl": "num_hidden_layers", "nh": "num_attention_heads", "final_norm": "model.norm"},
}
TYPE_MAP = {
    "gpt2": "gpt2", "bloom": "bloom", "qwen": "qwen", "qwen2": "qwen",
    "pythia": "pythia", "gpt-neox": "pythia", "aya": "aya", "command-r": "aya",
}


def infer_type(name):
    n = name.lower()
    for k, v in TYPE_MAP.items():
        if k in n:
            return v
    raise ValueError(f"cannot infer model type from {name}")


@dataclass(frozen=True, order=True)
class Head:
    layer: int
    head: int

    def __repr__(self):
        return f"L{self.layer}H{self.head}"


@contextmanager
def _null_ctx():
    yield


class GradModel:
    def __init__(self, name, load_path=None, model_type=None, load_in_4bit=False, attn_implementation=None):
        self.name = name
        self.mtype = model_type or infer_type(name)
        self.cfg = ARCH[self.mtype]
        self.is_quantized = load_in_4bit
        load_name = load_path or name

        if attn_implementation is None and self.mtype == "qwen":
            attn_implementation = "eager"
        self.attn_implementation = attn_implementation

        self.tok = AutoTokenizer.from_pretrained(load_name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

        kw = {"torch_dtype": config.DTYPE}
        if self.is_quantized:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=config.DTYPE,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        if self.attn_implementation is not None:
            kw["attn_implementation"] = self.attn_implementation

        self.model = AutoModelForCausalLM.from_pretrained(load_name, **kw)
        self.model.to(config.DEVICE)
        self.model.eval()

        cfg = self.model.config
        self.n_layers = getattr(cfg, self.cfg["nl"])
        self.n_heads = getattr(cfg, self.cfg["nh"])
        self.hidden = cfg.hidden_size if hasattr(cfg, "hidden_size") else cfg.n_embd
        self.head_dim = self.hidden // self.n_heads

    def _attn(self, layer):
        m = self.model
        for p in self.cfg["attn"].format(L=layer).split("."):
            m = getattr(m, p)
        return m

    def _block(self, layer):
        m = self.model
        for p in self.cfg["block"].format(L=layer).split("."):
            m = getattr(m, p)
        return m

    def _proj(self, layer):
        return getattr(self._attn(layer), self.cfg["proj"])

    def all_heads(self):
        return [Head(l, h) for l in range(self.n_layers) for h in range(self.n_heads)]

    def _proj_weight_info(self, layer):
        proj = self._proj(layer)
        if isinstance(proj, HFConv1D):
            return proj.weight, False
        W = proj.weight
        if hasattr(W, "quant_state"):
            import bitsandbytes as bnb
            return bnb.functional.dequantize_4bit(W.data, W.quant_state), True
        return W, True

    def head_slice_of(self, layer, x):
        bsz, seq_len, width = x.shape
        return x.reshape(bsz, seq_len, self.n_heads, self.head_dim)

    def _encode(self, prompt):
        inp = self.tok(prompt, return_tensors="pt")
        return {k: v.to(config.DEVICE) for k, v in inp.items()}

    @contextmanager
    def capture_hooks(self, store, requires_grad, retain_head_grad=True):
        handles = []
        for layer in range(self.n_layers):
            def make_hook(layer=layer):
                def fn(module, inp, output):
                    x = inp[0]
                    if x is None or x.dim() < 3:
                        return output
                    W, is_linear = self._proj_weight_info(layer)
                    xh_all = self.head_slice_of(layer, x)
                    contribs = []
                    for h in range(self.n_heads):
                        xh = xh_all[:, :, h, :]
                        if is_linear:
                            W_h = W[:, h * self.head_dim:(h + 1) * self.head_dim]
                            if xh.dtype != W_h.dtype:
                                xh = xh.to(W_h.dtype)
                            c = xh @ W_h.T
                        else:
                            W_h = W[h * self.head_dim:(h + 1) * self.head_dim, :]
                            if xh.dtype != W_h.dtype:
                                xh = xh.to(W_h.dtype)
                            c = xh @ W_h
                        c = c.to(config.DTYPE)
                        if requires_grad and retain_head_grad:
                            c.retain_grad()
                            store[Head(layer, h)] = c
                        else:
                            store[Head(layer, h)] = c[:, -1, :].detach().cpu() if c.dim() == 3 else c.detach().cpu()
                        contribs.append(c)
                    total = sum(contribs)
                    bias = getattr(module, "bias", None)
                    return total + bias if bias is not None else total
                return fn
            handles.append(self._proj(layer).register_forward_hook(make_hook()))
        try:
            yield
        finally:
            for h in handles:
                h.remove()

    @contextmanager
    def capture_dst_inputs(self, store):
        handles = []
        for layer in range(self.n_layers):
            def make_hook(layer=layer):
                def fn(module, args, kwargs):
                    x = kwargs.get("hidden_states")
                    if x is None and len(args) > 0:
                        x = args[0]
                    if x is None or not torch.is_tensor(x) or x.dim() < 2:
                        return None
                    x.retain_grad()
                    store[layer] = x
                    return None
                return fn
            handles.append(self._attn(layer).register_forward_pre_hook(make_hook(), with_kwargs=True))
        try:
            yield
        finally:
            for h in handles:
                h.remove()

    @contextmanager
    def capture_block_inputs(self, store):
        handles = []
        for layer in range(self.n_layers):
            def make_hook(layer=layer):
                def fn(module, args, kwargs):
                    x = kwargs.get("hidden_states")
                    if x is None and len(args) > 0:
                        x = args[0]
                    if x is None or not torch.is_tensor(x) or x.dim() < 2:
                        return None
                    x.retain_grad()
                    store[layer] = x
                    return None
                return fn
            handles.append(self._block(layer).register_forward_pre_hook(make_hook(), with_kwargs=True))
        try:
            yield
        finally:
            for h in handles:
                h.remove()

    def capture_preproj_heads(self, prompt):
        store = {}
        handles = []
        for layer in range(self.n_layers):
            def make_hook(layer=layer, _store=store):
                def fn(module, inp):
                    xh = self.head_slice_of(layer, inp[0])[:, -1, :, :].detach().clone()
                    for h in range(self.n_heads):
                        _store[Head(layer, h)] = xh[:, h, :]
                    return inp
                return fn
            handles.append(self._proj(layer).register_forward_pre_hook(make_hook()))
        try:
            with torch.no_grad():
                _ = self.first_token_logits(prompt)
        finally:
            for h in handles:
                h.remove()
        return store

    @contextmanager
    def patch_heads_ctx(self, patch_map):
        handles = []
        by_layer = {}
        for head, value in patch_map.items():
            by_layer.setdefault(head.layer, {})[head.head] = value
        for layer, head_values in by_layer.items():
            def hook_fn(module, inp, _layer=layer, _head_values=head_values):
                x = inp[0]
                xh = self.head_slice_of(_layer, x).clone()
                for h_idx, value in _head_values.items():
                    xh[:, -1, h_idx, :] = value.to(xh.dtype)
                return (xh.reshape(x.shape),) + tuple(inp[1:])
            handles.append(self._proj(layer).register_forward_pre_hook(hook_fn))
        try:
            yield
        finally:
            for h in handles:
                h.remove()

    @contextmanager
    def add_residual_ctx(self, layer, vector):
        def hook_fn(module, inp, output):
            if isinstance(output, tuple):
                hs = output[0]
                hs = hs.clone()
                hs[:, -1, :] = hs[:, -1, :] + vector.to(hs.dtype)
                return (hs,) + tuple(output[1:])
            hs = output.clone()
            hs[:, -1, :] = hs[:, -1, :] + vector.to(hs.dtype)
            return hs
        handle = self._block(layer).register_forward_hook(hook_fn)
        try:
            yield
        finally:
            handle.remove()

    def first_token_logits(self, prompt, store=None, requires_grad=False, retain_head_grad=True):
        inp = self._encode(prompt)
        ctx = self.capture_hooks(store, requires_grad, retain_head_grad) if store is not None else _null_ctx()
        if requires_grad:
            with ctx:
                out = self.model(**inp)
        else:
            with torch.no_grad(), ctx:
                out = self.model(**inp)
        return out.logits[:, -1, :]

    def hidden_states(self, prompt):
        inp = self._encode(prompt)
        with torch.no_grad():
            out = self.model(**inp, output_hidden_states=True)
        return out.hidden_states

    def _final_norm(self):
        m = self.model
        for p in self.cfg["final_norm"].split("."):
            m = getattr(m, p)
        return m

    def unembed(self, hidden_vec, apply_final_norm=True):
        if apply_final_norm:
            hidden_vec = self._final_norm()(hidden_vec)
        W = self.model.get_output_embeddings().weight
        return hidden_vec.to(W.dtype) @ W.T

    def free(self):
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
