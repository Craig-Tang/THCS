from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, MutableMapping, Set, Tuple

import pandas as pd


@dataclass(frozen=True)
class HalfStep:
    edge_type: int
    forward: bool


@dataclass(frozen=True)
class MetaPathSpec:
    edge_sequence: Tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.edge_sequence or len(self.edge_sequence) % 2 != 0:
            raise ValueError("MetaPath must be non-empty and symmetric (even length)")
        half = len(self.edge_sequence) // 2
        first_half = self.edge_sequence[:half]
        second_half = self.edge_sequence[half:]
        mirror = tuple(-x for x in reversed(first_half))
        if tuple(second_half) != mirror:
            raise ValueError("MetaPath not symmetric")

    def half_path_from_center(self) -> Tuple[HalfStep, ...]:
        half = len(self.edge_sequence) // 2
        second_half = self.edge_sequence[half:]
        return tuple(HalfStep(edge_type=abs(t), forward=(t > 0)) for t in second_half)


DirectionalAdjacency = Dict[Tuple[int, bool], Dict[int, Set[int]]]


def _as_int(value: object) -> int:
    try:
        return int(value)
    except Exception as exc:  # pragma: no cover
        raise TypeError(f"Expected int-like, got {value!r}") from exc


def build_directional_adjacency(edges: pd.DataFrame) -> DirectionalAdjacency:
    adjacency: DirectionalAdjacency = {}
    if edges.empty:
        return adjacency
    for edge_type, df_type in edges.groupby("edge_type", sort=False):
        fwd: Dict[int, Set[int]] = {}
        rev: Dict[int, Set[int]] = {}
        for src_id, group in df_type.groupby("src_id", sort=False):
            fwd[_as_int(src_id)] = {_as_int(d) for d in group["dst_id"]}
        for dst_id, group in df_type.groupby("dst_id", sort=False):
            rev[_as_int(dst_id)] = {_as_int(s) for s in group["src_id"]}
        et = _as_int(edge_type)
        adjacency[(et, True)] = fwd
        adjacency[(et, False)] = rev
    return adjacency


def add_edge_to_adjacency(adjacency: DirectionalAdjacency, edge_type: int, src: int, dst: int) -> None:
    et, s, d = _as_int(edge_type), _as_int(src), _as_int(dst)
    adjacency.setdefault((et, True), {}).setdefault(s, set()).add(d)
    adjacency.setdefault((et, False), {}).setdefault(d, set()).add(s)


def remove_edge_from_adjacency(adjacency: DirectionalAdjacency, edge_type: int, src: int, dst: int) -> None:
    et, s, d = _as_int(edge_type), _as_int(src), _as_int(dst)
    fwd = adjacency.get((et, True))
    rev = adjacency.get((et, False))
    if fwd is not None:
        nbrs = fwd.get(s)
        if nbrs is not None:
            nbrs.discard(d)
            if not nbrs:
                fwd.pop(s, None)
    if rev is not None:
        nbrs = rev.get(d)
        if nbrs is not None:
            nbrs.discard(s)
            if not nbrs:
                rev.pop(d, None)


def expand_once(frontier: Iterable[int], step: HalfStep, adjacency: DirectionalAdjacency) -> Set[int]:
    neighbor_map = adjacency.get((step.edge_type, step.forward))
    if not neighbor_map:
        return set()
    out: Set[int] = set()
    for u in frontier:
        nbrs = neighbor_map.get(_as_int(u))
        if nbrs:
            out.update(nbrs)
    return out


def k_core_nodes(graph: MutableMapping[int, Set[int]], k: int) -> Tuple[int, ...]:
    if k <= 0:
        return tuple(sorted(graph.keys()))
    if not graph:
        return tuple()
    adj: Dict[int, Set[int]] = {u: set(vs) for u, vs in graph.items()}
    queue = deque([u for u, vs in adj.items() if len(vs) < k])
    removed: Set[int] = set()
    while queue:
        u = queue.popleft()
        if u in removed:
            continue
        removed.add(u)
        for v in list(adj.get(u, [])):
            adj[v].discard(u)
            if len(adj[v]) < k and v not in removed:
                queue.append(v)
        adj.pop(u, None)
    return tuple(sorted(adj.keys()))
