from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure local package import (self-contained)
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from thcs.online import QueryParams, run_csw, run_ice
from thcs.index import TemporalCoreIndex, load_inverted_index_json, query_inverted_index


def parse_meta_path(s: str) -> tuple[int, ...]:
    s = s.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    parts = [p for p in s.replace(" ", "").split(",") if p]
    return tuple(int(p) for p in parts)


def cmd_search(args: argparse.Namespace) -> None:
    q = QueryParams(
        k=args.k,
        delta=args.delta,
        time_interval=(int(args.t_start), int(args.t_end)),
        meta_path=args.meta_path,
    )
    if args.mode == "ice":
        r = run_ice(args.source, q)
    else:
        r = run_csw(args.source, q)
    nodes_list = list(r.core_nodes)
    print({"nodes": nodes_list[:50], "result_size": len(nodes_list), "stats": r.stats})


def cmd_index_build(args: argparse.Namespace) -> None:
    time_range = None
    if args.t_start is not None and args.t_end is not None:
        time_range = (int(args.t_start), int(args.t_end))
    tci = TemporalCoreIndex.from_source(args.source, args.meta_path, time_range=time_range)
    stats = tci.build()
    inv = tci.to_inverted()
    out = args.out
    if out is None:
        idx_dir = THIS_DIR / "idx"
        idx_dir.mkdir(parents=True, exist_ok=True)
        src_path = Path(args.source)
        if src_path.suffix.lower() == ".parquet":
            src_name = src_path.parent.name or src_path.stem
        else:
            src_name = src_path.name
        mp = "_".join(str(x) for x in args.meta_path)
        if time_range:
            out = idx_dir / f"{src_name}_{mp}_{time_range[0]}_{time_range[1]}_index.json"
        else:
            out = idx_dir / f"{src_name}_{mp}_full_index.json"
    with Path(out).open("w", encoding="utf-8") as f:
        json.dump(inv, f)
    print({"windows": stats.windows, "edges_scanned": stats.total_edges_scanned, "written": str(out)})


def cmd_index_query(args: argparse.Namespace) -> None:
    inv = load_inverted_index_json(args.index_json)
    nodes = query_inverted_index(inv, k=args.k, delta=args.delta, T_q=(args.t_start, args.t_end))
    out = {"result_size": len(nodes), "nodes": list(nodes)[:50]}
    print(out)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="THIN Community Search - Formal Release")
    sub = p.add_subparsers(dest="cmd", required=True)

    # search
    sp = sub.add_parser("search", help="online search: csw/ice")
    sp.add_argument("source")
    sp.add_argument("--meta-path", type=parse_meta_path, required=True)
    sp.add_argument("--k", type=int, required=True)
    sp.add_argument("--delta", type=int, required=True)
    sp.add_argument("--t-start", type=int, required=True)
    sp.add_argument("--t-end", type=int, required=True)
    sp.add_argument("--mode", choices=["csw", "ice"], default="csw")
    sp.set_defaults(func=cmd_search)

    # index-build
    ib = sub.add_parser("index-build", help="build index and export JSON")
    ib.add_argument("source")
    ib.add_argument("--meta-path", type=parse_meta_path, required=True)
    ib.add_argument("--t-start", type=int)
    ib.add_argument("--t-end", type=int)
    ib.add_argument("--out", type=Path)
    ib.set_defaults(func=cmd_index_build)

    # index-query
    iq = sub.add_parser("index-query", help="query index JSON")
    iq.add_argument("index_json", type=Path)
    iq.add_argument("--k", type=int, required=True)
    iq.add_argument("--delta", type=int, required=True)
    iq.add_argument("--t-start", type=int, required=True)
    iq.add_argument("--t-end", type=int, required=True)
    iq.set_defaults(func=cmd_index_query)

    return p


def main() -> None:
    p = build_parser()
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
