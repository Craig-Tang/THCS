from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Union, Iterable as _Iterable
import time

import pandas as pd

from . import graph_builder

# Base dir points to repo root (one level above this package)
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

@dataclass(frozen=True)
class QueryParams:
    k: int
    delta: int
    time_interval: Tuple[int, int]
    meta_path: Tuple[int, ...]
    meta_path_name: Optional[str] = None


@dataclass(frozen=True)
class QueryResult:
    core_nodes: Iterable[int]
    stats: Dict[str, object]


EdgeSource = Union[pd.DataFrame, str, Path]


def run_csw(
    edges: EdgeSource,
    query: QueryParams,
    schema: Optional[Dict] = None,
) -> QueryResult:
    t0 = time.perf_counter()
    if not query.meta_path:
        raise ValueError("meta_path required")

    edge_types = {abs(edge_type) for edge_type in query.meta_path}
    edges_df, _ = _prepare_edges(
        edges, schema, edge_types=edge_types, time_interval=query.time_interval
    )

    meta_spec = graph_builder.MetaPathSpec(tuple(query.meta_path))
    half_path = meta_spec.half_path_from_center()

    window_iter = _iter_windows(edges_df, query.time_interval, query.delta)
    derived_graph: Dict[int, set[int]] = {}
    windows_processed = 0
    edges_scanned = 0

    for window_start, window_end, window_edges in window_iter:
        if window_edges.empty:
            continue
        windows_processed += 1
        edges_scanned += int(len(window_edges))
        adjacency = graph_builder.build_directional_adjacency(window_edges)
        _process_window(half_path, adjacency, derived_graph)

    core_nodes = graph_builder.k_core_nodes(derived_graph, query.k)

    runtime = time.perf_counter() - t0
    stats: Dict[str, object] = {
        "windows": float(windows_processed),
        "edges_scanned": float(edges_scanned),
        "result_size": float(len(core_nodes)),
        "runtime_sec": float(runtime),
    }
    return QueryResult(core_nodes=core_nodes, stats=stats)


def run_ice(
    edges: EdgeSource,
    query: QueryParams,
    schema: Optional[Dict] = None,
) -> QueryResult:
    t0 = time.perf_counter()
    if not query.meta_path:
        raise ValueError("meta_path required")

    edge_types = {abs(t) for t in query.meta_path}
    edges_df, _ = _prepare_edges(
        edges, schema, edge_types=edge_types, time_interval=query.time_interval
    )

    meta_spec = graph_builder.MetaPathSpec(tuple(query.meta_path))
    half_path = meta_spec.half_path_from_center()

    t_start, t_end = query.time_interval
    buckets: Dict[int, pd.DataFrame] = {}
    for t, df in edges_df.groupby("timestamp", sort=True):
        ti = _as_int(t)
        if t_start <= ti <= t_end:
            buckets[ti] = df

    first_start = t_start
    first_end = min(t_end, first_start + query.delta)
    current_edges = pd.concat(
        [buckets[t] for t in range(first_start, first_end + 1) if t in buckets],
        ignore_index=True,
    ) if buckets else pd.DataFrame(columns=edges_df.columns)

    adjacency = graph_builder.build_directional_adjacency(current_edges)

    derived_graph: Dict[int, set[int]] = {}
    windows_processed = 0
    edges_scanned = int(len(current_edges))

    def expand_centers(_adj: Dict[Tuple[int, bool], Dict[int, set[int]]]) -> None:
        if not half_path:
            return
        first = half_path[0]
        centers = _adj.get((first.edge_type, first.forward), {})
        for _, initial_targets in centers.items():
            current_nodes = set(initial_targets)
            for step in half_path[1:]:
                current_nodes = graph_builder.expand_once(current_nodes, step, _adj)
                if not current_nodes:
                    break
            if current_nodes:
                _add_clique_to_graph(current_nodes, derived_graph)

    expand_centers(adjacency)
    windows_processed += 1

    total_windows = max(0, (t_end - t_start) - query.delta + 1)
    for w in range(total_windows):
        win_start = first_start + w
        win_end = win_start + query.delta
        next_end = min(t_end, win_end + 1)

        if win_start in buckets:
            for row in buckets[win_start].itertuples(index=False):
                graph_builder.remove_edge_from_adjacency(
                    adjacency, getattr(row, "edge_type"), getattr(row, "src_id"), getattr(row, "dst_id")
                )
        if next_end in buckets:
            df_add = buckets[next_end]
            edges_scanned += int(len(df_add))
            for row in df_add.itertuples(index=False):
                graph_builder.add_edge_to_adjacency(
                    adjacency, getattr(row, "edge_type"), getattr(row, "src_id"), getattr(row, "dst_id")
                )
        expand_centers(adjacency)
        windows_processed += 1

    core_nodes = graph_builder.k_core_nodes(derived_graph, query.k)
    runtime = time.perf_counter() - t0
    stats: Dict[str, object] = {
        "windows": float(windows_processed),
        "edges_scanned": float(edges_scanned),
        "result_size": float(len(core_nodes)),
        "mode": "ICE",
        "runtime_sec": float(runtime),
    }
    return QueryResult(core_nodes=core_nodes, stats=stats)


def _iter_windows(edges: pd.DataFrame, time_interval: Tuple[int, int], delta: int):
    start, end = time_interval
    window_count = (end - start) - delta + 2
    if window_count <= 0:
        window_count = 0
    for offset in range(window_count):
        window_start = start + offset
        window_end = window_start + delta
        mask = (edges["timestamp"] >= window_start) & (edges["timestamp"] <= window_end)
        yield window_start, window_end, edges.loc[mask]


def _prepare_edges(
    source: EdgeSource,
    schema: Optional[Dict],
    *,
    edge_types: Optional[_Iterable[int]] = None,
    time_interval: Optional[Tuple[int, int]] = None,
) -> Tuple[pd.DataFrame, Dict]:
    if isinstance(source, pd.DataFrame):
        edges_df = source.copy()
        edges_df = _filter_edges_df(edges_df, edge_types=edge_types, time_interval=time_interval)
        inferred = _infer_schema(edges_df)
        return edges_df, schema or inferred

    edges_path = _resolve_edges_path(source)
    filters = _build_parquet_filters(edge_types=edge_types, time_interval=time_interval)
    try:
        edges_df = pd.read_parquet(edges_path, filters=filters)
    except TypeError:
        edges_df = pd.read_parquet(edges_path)
    edges_df = _filter_edges_df(edges_df, edge_types=edge_types, time_interval=time_interval)
    inferred = _infer_schema(edges_df, edges_path)
    return edges_df, schema or inferred


def _resolve_edges_path(source: Union[str, Path]) -> Path:
    candidate = Path(source)
    # 1) If a directory is provided, expect edges.parquet inside it
    if candidate.is_dir():
        candidate = candidate / "edges.parquet"
    else:
    # 2) If no suffix (dataset name or folder-like), try data/<dataset>/edges.parquet
        if not candidate.suffix:
            dataset_dir = DATA_DIR / str(source)
            if dataset_dir.exists() and dataset_dir.is_dir():
                candidate = dataset_dir / "edges.parquet"
    # Now require a .parquet file
    if candidate.suffix.lower() != ".parquet":
        raise ValueError(
            f"Unsupported edges source: {source!r}. Provide a dataset name under 'data/', a directory containing 'edges.parquet', or a .parquet file path."
        )
    if not candidate.exists():
        raise FileNotFoundError(f"Edges parquet not found at {candidate}")
    return candidate


def _infer_schema(edges: pd.DataFrame, edges_path: Optional[Path] = None) -> Dict:
    schema: Dict[str, Dict[int, str]] = {}
    if "edge_type" in edges.columns and "type_label" in edges.columns:
        mapping = (
            edges[["edge_type", "type_label"]].drop_duplicates().dropna().itertuples(index=False)
        )
        schema["edge_type_labels"] = { _as_int(m.edge_type): str(m.type_label) for m in mapping }
    if edges_path is not None:
        nodes_path = edges_path.with_name("nodes.parquet")
        if nodes_path.exists():
            nodes_df = pd.read_parquet(nodes_path, columns=["node_type", "type_label"])
            mapping = nodes_df.drop_duplicates(subset=["node_type", "type_label"]).dropna().itertuples(index=False)
            schema["node_type_labels"] = { _as_int(m.node_type): str(m.type_label) for m in mapping }
    return schema


def _as_int(value: object) -> int:
    try:
        return int(value)
    except Exception as exc:  # pragma: no cover
        raise TypeError(f"Expected int-like value, got {value!r}") from exc


def _filter_edges_df(
    edges: pd.DataFrame,
    *,
    edge_types: Optional[_Iterable[int]],
    time_interval: Optional[Tuple[int, int]],
) -> pd.DataFrame:
    filtered = edges
    if edge_types is not None and "edge_type" in filtered.columns:
        allowed = {int(t) for t in edge_types}
        filtered = filtered[filtered["edge_type"].isin(allowed)]
    if time_interval is not None and "timestamp" in filtered.columns:
        start, end = time_interval
        filtered = filtered[(filtered["timestamp"] >= start) & (filtered["timestamp"] <= end)]
    return filtered


def _build_parquet_filters(*, edge_types: Optional[_Iterable[int]], time_interval: Optional[Tuple[int, int]]):
    predicates = []
    if edge_types:
        predicates.append(("edge_type", "in", [int(t) for t in edge_types]))
    if time_interval:
        start, end = time_interval
        predicates.append(("timestamp", ">=", int(start)))
        predicates.append(("timestamp", "<=", int(end)))
    return predicates or None


def _process_window(
    half_path: Tuple[graph_builder.HalfStep, ...],
    adjacency: Dict[Tuple[int, bool], Dict[int, set[int]]],
    derived_graph: Dict[int, set[int]],
) -> None:
    if not half_path:
        return
    first = half_path[0]
    centers = adjacency.get((first.edge_type, first.forward), {})
    if not centers:
        return
    for _, initial_targets in centers.items():
        curr = set(initial_targets)
        for step in half_path[1:]:
            curr = graph_builder.expand_once(curr, step, adjacency)
            if not curr:
                break
        if curr:
            _add_clique_to_graph(curr, derived_graph)


def _add_clique_to_graph(nodes: Iterable[int], graph: Dict[int, set[int]]) -> None:
    arr = list(nodes)
    n = len(arr)
    for i in range(n):
        u = arr[i]
        graph.setdefault(u, set())
        for j in range(i + 1, n):
            v = arr[j]
            graph.setdefault(v, set())
            graph[u].add(v)
            graph[v].add(u)
