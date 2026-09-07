"""Deterministic structured attention inputs shared by correctness tests."""

from __future__ import annotations

import mlx.core as mx


def structured_qkv(
    *,
    length: int = 2_048,
    query_heads: int = 8,
    kv_heads: int = 2,
    head_dim: int = 64,
    block_size: int = 128,
    clusters: int = 16,
    noise: float = 0.3,
    dtype: mx.Dtype = mx.float32,
    seed: int = 0,
) -> tuple[mx.array, mx.array, mx.array]:
    """Return clustered ``q, k, v`` arrays shaped ``[heads, length, dim]``."""
    if length <= 0 or head_dim <= 0 or block_size <= 0 or clusters <= 0:
        raise ValueError("length, head_dim, block_size, and clusters must be positive")
    if query_heads <= 0 or kv_heads <= 0 or query_heads % kv_heads:
        raise ValueError("query_heads must be divisible by positive kv_heads")

    mx.random.seed(seed)
    centers = 1.2 * mx.random.normal((kv_heads, clusters, head_dim)).astype(mx.float32)

    num_blocks = (length + block_size - 1) // block_size
    kv_block_clusters = mx.random.randint(
        0, clusters, shape=(kv_heads, num_blocks)
    )
    block_ids = mx.arange(length, dtype=mx.int32) // block_size
    kv_cluster_ids = kv_block_clusters[:, block_ids]
    head_ids = mx.arange(kv_heads, dtype=mx.int32)[:, None]
    keys = centers[head_ids, kv_cluster_ids]
    keys = keys + noise * mx.random.normal(keys.shape)

    group_size = query_heads // kv_heads
    q_kv_ids = mx.arange(query_heads, dtype=mx.int32) // group_size
    query_run = max(1, block_size // 8)
    query_run_ids = mx.arange(length, dtype=mx.int32) // query_run
    q_run_clusters = mx.random.randint(
        0,
        clusters,
        shape=(query_heads, (length + query_run - 1) // query_run),
    )
    q_cluster_ids = q_run_clusters[:, query_run_ids]
    queries = centers[q_kv_ids[:, None], q_cluster_ids]
    queries = queries + noise * mx.random.normal(queries.shape)
    values = mx.random.normal((kv_heads, length, head_dim)).astype(mx.float32)
    return queries.astype(dtype), keys.astype(dtype), values.astype(dtype)
