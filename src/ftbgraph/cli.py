import argparse
import json

from .pipeline import run_novel_pipeline
from . import config


def main():
    parser = argparse.ArgumentParser(prog="ftbgraph")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run")
    run_p.add_argument("--model", required=True)
    run_p.add_argument("--type", required=True)
    run_p.add_argument("--load-path", default=None)
    run_p.add_argument("--4bit", dest="load_in_4bit", action="store_true")
    run_p.add_argument("--eap-n", type=int, default=24)
    run_p.add_argument("--verify-topk", type=int, default=40)
    run_p.add_argument("--verify-dst-cap", type=int, default=4)
    run_p.add_argument("--verify-n", type=int, default=20)
    run_p.add_argument("--ig-steps", type=int, default=6)
    run_p.add_argument("--valid-n", type=int, default=24)
    run_p.add_argument("--pivot-n", type=int, default=20)
    run_p.add_argument("--no-fdr", dest="require_fdr", action="store_false")
    run_p.add_argument("--out-dir", default=None)

    args = parser.parse_args()
    if args.command == "run":
        result = run_novel_pipeline(
            args.model, args.type, load_path=args.load_path, load_in_4bit=args.load_in_4bit,
            eap_n=args.eap_n, verify_topk=args.verify_topk, verify_dst_cap=args.verify_dst_cap,
            verify_n=args.verify_n, ig_steps=args.ig_steps, valid_n=args.valid_n, pivot_n=args.pivot_n,
            require_fdr=args.require_fdr, out_dir=args.out_dir,
        )
        print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
