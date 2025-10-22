from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd

from . import graph_builder
from .online import _prepare_edges, _as_int


def center_target_incidence(
    adjacency: Dict[Tuple[int, bool], Dict[int, Set[int]]],
    half_path: Tuple[graph_builder.HalfStep, ...],
) -> Dict[int, Set[int]]:
    out: Dict[int, Set[int]] = {}
    if not half_path:
        return out
    first = half_path[0]
    centers = adjacency.get((first.edge_type, first.forward), {})
    for cnode, init_targets in centers.items():
        curr = set(init_targets)
        for step in half_path[1:]:
            curr = graph_builder.expand_once(curr, step, adjacency)
            if not curr:
                break
        if curr:
            out[_as_int(cnode)] = {_as_int(t) for t in curr}
    return out


def build_target_graph_boolean(M: Dict[int, Set[int]]) -> Dict[int, Set[int]]:
    G: Dict[int, Set[int]] = {}
    for targets in M.values():
        arr = list(targets)
        n = len(arr)
        for i in range(n):
            u = arr[i]
            G.setdefault(u, set())
            for j in range(i + 1, n):
                v = arr[j]
                G.setdefault(v, set())
                G[u].add(v)
                G[v].add(u)
    return G


def core_numbers(graph: Dict[int, Set[int]]) -> Dict[int, int]:
    if not graph:
        return {}
    adj: Dict[int, Set[int]] = {u: set(vs) for u, vs in graph.items()}
    nodes = list(adj.keys())
    n = len(nodes)
    deg: Dict[int, int] = {u: len(adj[u]) for u in nodes}
    if n == 0:
        return {}
    max_deg = max(deg.values(), default=0)
    bin_count = [0] * (max_deg + 1)
    for u in nodes:
        bin_count[deg[u]] += 1
    start = [0] * (max_deg + 1)
    s = 0
    for d in range(max_deg + 1):
        start[d] = s
        s += bin_count[d]
    vert = [0] * n
    pos: Dict[int, int] = {}
    next_idx = start[:]
    for u in nodes:
        d = deg[u]
        pos[u] = next_idx[d]
        vert[pos[u]] = u
        next_idx[d] += 1
    core: Dict[int, int] = {}
    for i in range(n):
        u = vert[i]
        core[u] = deg[u]
        for v in adj[u]:
            if deg[v] > deg[u]:
                dv = deg[v]
                pv = pos[v]
                pw = start[dv]
                w = vert[pw]
                if v != w:
                    vert[pv], vert[pw] = vert[pw], vert[pv]
                    pos[v], pos[w] = pw, pv
                start[dv] += 1
                deg[v] -= 1
    return core


@dataclass
class IndexBuildStats:
    windows: int
    runtime_sec: float
    total_edges_scanned: int


class TemporalCoreIndex:
    def __init__(self, edges: pd.DataFrame, meta_path: Tuple[int, ...]):
        self.edges = edges
        self.meta_spec = graph_builder.MetaPathSpec(tuple(meta_path))
        self.half_path = self.meta_spec.half_path_from_center()
        self.node_index: Dict[int, Dict[int, Dict[int, int]]] = {}

    @classmethod
    def from_source(
        cls,
        source: pd.DataFrame | str | Path,
        meta_path: Tuple[int, ...],
        *,
        time_range: Optional[Tuple[int, int]] = None,
    ) -> "TemporalCoreIndex":
        edge_types = {abs(t) for t in meta_path}
        edges_df, _ = _prepare_edges(
            source, None,
            edge_types=edge_types, time_interval=time_range,
        )
        return cls(edges_df, tuple(meta_path))

    def build(self) -> IndexBuildStats:
        import time as _time
        t0 = _time.perf_counter()
        if self.edges.empty:
            return IndexBuildStats(0, 0.0, 0)
        t_min = int(self.edges["timestamp"].min())
        t_max = int(self.edges["timestamp"].max())
        buckets: Dict[int, pd.DataFrame] = { _as_int(t): df for t, df in self.edges.groupby("timestamp", sort=True)}
        windows = 0
        total_scanned = 0
        for t_start in range(t_min, t_max + 1):
            current_edges = buckets.get(t_start)
            if current_edges is None:
                current_edges = pd.DataFrame(columns=self.edges.columns)
            adjacency = graph_builder.build_directional_adjacency(current_edges)
            M_cache = center_target_incidence(adjacency, self.half_path)
            G_cache = build_target_graph_boolean(M_cache)
            cores = core_numbers(G_cache)
            self._record_interval_batch(cores, t_start, t_start)
            windows += 1
            for t_end in range(t_start + 1, t_max + 1):
                df_add = buckets.get(t_end)
                if df_add is not None and not df_add.empty:
                    total_scanned += int(len(df_add))
                    for row in df_add.itertuples(index=False):
                        graph_builder.add_edge_to_adjacency(adjacency, getattr(row, "edge_type"), getattr(row, "src_id"), getattr(row, "dst_id"))
                    delta_M = self._delta_center_targets(df_add, self.half_path, adjacency)
                    self._apply_delta(M_cache, G_cache, delta_M)
                cores = core_numbers(G_cache)
                self._record_interval_batch(cores, t_start, t_end)
                windows += 1
        runtime = _time.perf_counter() - t0
        return IndexBuildStats(windows=windows, runtime_sec=float(runtime), total_edges_scanned=total_scanned)

    def _record_interval_batch(self, cores: Dict[int, int], t_start: int, t_end: int) -> None:
        for node, cmax in cores.items():
            if cmax <= 0:
                continue
            entry = self.node_index.setdefault(int(node), {})
            for c in range(1, cmax + 1):
                entry.setdefault(c, {})[int(t_end)] = int(t_start)

    def to_inverted(self) -> Dict[str, Dict[str, List[int]]]:
        inv: Dict[str, Dict[str, List[int]]] = {}
        for node, core_map in self.node_index.items():
            for core, tend_map in core_map.items():
                out = inv.setdefault(str(int(core)), {})
                for t_end, t_start in tend_map.items():
                    key = f"{int(t_start)}_{int(t_end)}"
                    out.setdefault(key, []).append(int(node))
        for out in inv.values():
            for k in out:
                out[k] = sorted(set(out[k]))
        return inv

    def query(self, k: int, delta: int, T_q: Tuple[int, int]) -> Set[int]:
        return query_inverted_index(self.to_inverted(), k=k, delta=delta, T_q=T_q)

    def _reverse_step(self, step: graph_builder.HalfStep) -> graph_builder.HalfStep:
        return graph_builder.HalfStep(edge_type=step.edge_type, forward=(not step.forward))

    def _expand_forward(self, frontier: Set[int], steps: List[graph_builder.HalfStep], adjacency: Dict[Tuple[int, bool], Dict[int, Set[int]]]) -> Set[int]:
        curr = set(frontier)
        for st in steps:
            if not curr:
                break
            curr = graph_builder.expand_once(curr, st, adjacency)
        return curr

    def _expand_backward(self, frontier: Set[int], steps: List[graph_builder.HalfStep], adjacency: Dict[Tuple[int, bool], Dict[int, Set[int]]]) -> Set[int]:
        curr = set(frontier)
        for st in reversed(steps):
            if not curr:
                break
            curr = graph_builder.expand_once(curr, self._reverse_step(st), adjacency)
        return curr

    def _delta_center_targets(self, new_edges: pd.DataFrame, half_path: Tuple[graph_builder.HalfStep, ...], adjacency: Dict[Tuple[int, bool], Dict[int, Set[int]]]) -> Dict[int, Set[int]]:
        delta: Dict[int, Set[int]] = {}
        if new_edges is None or new_edges.empty or not half_path:
            return delta
        steps = list(half_path)
        L = len(steps)
        prefixes = [steps[:i] for i in range(L)]
        suffixes = [steps[i + 1 :] for i in range(L)]
        for row in new_edges.itertuples(index=False):
            et = _as_int(getattr(row, "edge_type"))
            src = _as_int(getattr(row, "src_id"))
            dst = _as_int(getattr(row, "dst_id"))
            for i, st in enumerate(steps):
                if st.edge_type != et:
                    continue
                u, v = (src, dst) if st.forward else (dst, src)
                C = {u} if i == 0 else self._expand_backward({u}, prefixes[i], adjacency)
                T = {v} if i == L - 1 else self._expand_forward({v}, suffixes[i], adjacency)
                if not C or not T:
                    continue
                for c in C:
                    bucket = delta.setdefault(int(c), set())
                    bucket.update(int(t) for t in T)
        return delta

    def _apply_delta(self, M_cache: Dict[int, Set[int]], G_cache: Dict[int, Set[int]], delta_M: Dict[int, Set[int]]) -> None:
        for c, newT in delta_M.items():
            if not newT:
                continue
            oldT = M_cache.get(c, set())
            arr = list(newT)
            n = len(arr)
            for i in range(n):
                u = arr[i]
                G_cache.setdefault(u, set())
                for j in range(i + 1, n):
                    v = arr[j]
                    G_cache.setdefault(v, set())
                    G_cache[u].add(v)
                    G_cache[v].add(u)
            if oldT:
                for u in newT:
                    G_cache.setdefault(u, set())
                    for v in oldT:
                        if u == v:
                            continue
                        G_cache.setdefault(v, set())
                        G_cache[u].add(v)
                        G_cache[v].add(u)
            if c in M_cache:
                M_cache[c].update(newT)
            else:
                M_cache[c] = set(newT)


def _normalize_inverted(inverted_index: Dict | None) -> Dict[int, Dict[Tuple[int, int], Set[int]]]:
    norm: Dict[int, Dict[Tuple[int, int], Set[int]]] = {}
    if not inverted_index:
        return norm
    for core_key, interval_map in inverted_index.items():
        core = int(core_key)
        out_inner = norm.setdefault(core, {})
        for ik, nodes in (interval_map or {}).items():
            if isinstance(ik, str) and "_" in ik:
                a, b = ik.split("_", 1)
                t_s, t_e = int(a), int(b)
            else:
                t_s, t_e = int(ik[0]), int(ik[1])
            out_inner[(t_s, t_e)] = {int(n) for n in nodes}
    return norm


def load_inverted_index_json(path: str | Path) -> Dict[int, Dict[Tuple[int, int], Set[int]]]:
    import json as _json
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = _json.load(f)
    return _normalize_inverted(data)


def query_inverted_index(inverted_index: Dict, *, k: int, delta: int, T_q: Tuple[int, int]) -> Set[int]:
    q_s, q_e = int(T_q[0]), int(T_q[1])
    result: Set[int] = set()
    inv = _normalize_inverted(inverted_index)
    for core, interval_dict in inv.items():
        if core < k:
            continue
        for (t_s, t_e), nodes in interval_dict.items():
            if t_s >= q_s and t_e <= q_e and (t_e - t_s) <= delta:
                result.update(nodes)
    return result
