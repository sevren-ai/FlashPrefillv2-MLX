"""Model-independent, conservative hook for MLX scaled-dot-product attention.

``enable`` replaces MLX's SDPA entry point with a wrapper but sends calls to
FPV2 only when they are batch-one, bottom-right causal prefill with matching
float16 or bfloat16 Q/K/V, supported head dimensions, grouped-query-compatible
head counts, at least one full query tile, and a key length meeting the
configured ``min_tokens`` gate. Array masks, attention sinks, extra SDPA
options, decode calls, and unsupported layouts are passed to the exact original
function unchanged.

For eligible calls, ``offset = key_length - query_length`` locates the query
chunk in the causal key prefix. Queries and keys are padded only for tile/block
processing; selection and sparse attention retain the original lengths and the
output padding is removed. ``disable`` restores the original MLX function.

This remains model-independent because full causal transformer layers pass the
string mask ``"causal"``; sliding-window layers pass array masks, SSM and linear
attention layers do not call SDPA, and quantized caches bypass ``mx.fast``.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import mlx.core as mx

from fpv2.kernel import SUPPORTED_HEAD_DIMS, compact_block_indices, sparse_attention
from fpv2.selection import FPV2Config, block_scores_and_select, pool_blocks


_ORIGINAL_SDPA = mx.fast.scaled_dot_product_attention
_enabled = False
_collect_stats = False
_selected_counts: list[mx.array] = []
_selected_dense_blocks = 0
_visible_blocks = 0
_HOPPER_PACKED_TILE_ROWS = 128


def _effective_config(
    config: FPV2Config, query_heads: int, kv_heads: int
) -> FPV2Config | None:
    """Resolve Hopper's default 128 packed rows into query positions."""
    if config.tile_q != _HOPPER_PACKED_TILE_ROWS:
        return config
    if not kv_heads or query_heads % kv_heads:
        return None
    group_size = query_heads // kv_heads
    if _HOPPER_PACKED_TILE_ROWS % group_size:
        return None
    return replace(config, tile_q=_HOPPER_PACKED_TILE_ROWS // group_size)


def _request_config(
    config: FPV2Config, key_length: int, total_tokens: int | None
) -> FPV2Config:
    """Apply the dense tail only when this call reaches the request boundary."""
    if config.last_n_blocks and (
        total_tokens is None or key_length < total_tokens
    ):
        return replace(config, last_n_blocks=0)
    return config


def _causal_block_count(
    kv_heads: int,
    query_length: int,
    offset: int,
    config: FPV2Config,
) -> int:
    count = 0
    for start in range(0, query_length, config.tile_q):
        end = offset + min(start + config.tile_q, query_length)
        count += (end + config.block_k - 1) // config.block_k
    return count * kv_heads


def block_sparsity() -> float:
    """Return the fraction of visible causal blocks skipped since ``enable``."""
    if not _visible_blocks:
        return 0.0
    selected = _selected_dense_blocks
    if _selected_counts:
        sparse_selected = mx.sum(mx.stack([mx.sum(x) for x in _selected_counts]))
        mx.eval(sparse_selected)
        selected += sparse_selected.item()
    return 1.0 - selected / _visible_blocks


def _pad_sequence(x: mx.array, length: int) -> mx.array:
    amount = length - x.shape[-2]
    if amount < 0:
        raise ValueError("target length cannot be shorter than the input")
    if not amount:
        return x
    widths = [(0, 0)] * x.ndim
    widths[-2] = (0, amount)
    return mx.pad(x, widths)


def fpv2_sdpa(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    *,
    scale: float,
    config: FPV2Config = FPV2Config(),
) -> mx.array:
    """Apply FPV2 to an eligible batch-one, bottom-right causal attention."""
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must have shape [batch, heads, sequence, dim]")
    if q.shape[0] != 1 or k.shape[0] != 1 or v.shape[0] != 1:
        raise ValueError("FPV2 currently supports batch size 1")
    effective_config = _effective_config(config, q.shape[1], k.shape[1])
    if effective_config is None:
        raise ValueError("GQA group size must divide 128 packed query rows")
    config = effective_config
    query_length = q.shape[-2]
    key_length = k.shape[-2]
    if query_length > key_length:
        raise ValueError("query length cannot exceed key length for causal attention")
    offset = key_length - query_length

    padded_query_length = (
        (query_length + config.tile_q - 1) // config.tile_q * config.tile_q
    )
    required_key_length = max(key_length, offset + padded_query_length)
    padded_key_length = (
        (required_key_length + config.block_k - 1)
        // config.block_k
        * config.block_k
    )
    padded_q = _pad_sequence(q[0], padded_query_length)
    padded_k = _pad_sequence(k[0], padded_key_length)
    padded_v = _pad_sequence(v[0], padded_key_length)
    pooled_k = pool_blocks(padded_k, config.block_k)
    pooled_v = pool_blocks(padded_v, config.block_k)
    selected, _ = block_scores_and_select(
        padded_q,
        pooled_k,
        scale=scale,
        config=config,
        offset=offset,
        query_length=query_length,
    )
    indices = compact_block_indices(
        selected,
        config=config,
        offset=offset,
        query_length=query_length,
        key_length=key_length,
    )
    if _collect_stats:
        global _visible_blocks
        _selected_counts.append(indices[1])
        _visible_blocks += _causal_block_count(
            k.shape[1], query_length, offset, config
        )
    output = sparse_attention(
        padded_q,
        padded_k,
        padded_v,
        pooled_k,
        pooled_v,
        *indices,
        scale=scale,
        config=config,
        offset=offset,
        query_length=query_length,
        key_length=key_length,
        pack_gqa=config.pack_gqa,
    )
    return output[None, :, :query_length, :]


def _eligible(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    mask: object,
    sinks: object,
    kwargs: dict[str, Any],
    config: FPV2Config,
    min_tokens: int,
) -> bool:
    if not isinstance(mask, str) or mask != "causal" or sinks is not None or kwargs:
        return False
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        return False
    if q.shape[0] != 1 or k.shape[0] != 1 or v.shape[0] != 1:
        return False
    if k.shape != v.shape or q.shape[-1] != k.shape[-1]:
        return False
    if q.shape[-2] > k.shape[-2] or q.shape[-2] < config.tile_q:
        return False
    if k.shape[-2] < min_tokens:
        return False
    if q.shape[-1] not in SUPPORTED_HEAD_DIMS:
        return False
    if q.dtype not in (mx.float16, mx.bfloat16):
        return False
    if k.dtype != q.dtype or v.dtype != q.dtype:
        return False
    return bool(k.shape[1] and q.shape[1] % k.shape[1] == 0)


def enable(
    config: FPV2Config | None = None,
    *,
    min_tokens: int = 4_096,
    collect_stats: bool = False,
    total_tokens: int | None = None,
    **config_fields: Any,
) -> None:
    """Replace MLX SDPA with an FPV2-gated, request-aware wrapper."""
    global _collect_stats, _enabled, _selected_dense_blocks, _visible_blocks
    if min_tokens <= 0:
        raise ValueError("min_tokens must be positive")
    if not isinstance(collect_stats, bool):
        raise ValueError("collect_stats must be a boolean")
    if total_tokens is not None and (
        not isinstance(total_tokens, int)
        or isinstance(total_tokens, bool)
        or total_tokens <= 0
    ):
        raise ValueError("total_tokens must be a positive integer or None")
    if config is not None and config_fields:
        config = replace(config, **config_fields)
    elif config is None:
        config = FPV2Config(**config_fields)
    _collect_stats = collect_stats
    _selected_counts.clear()
    _selected_dense_blocks = 0
    _visible_blocks = 0

    def wrapped(
        q: mx.array,
        k: mx.array,
        v: mx.array,
        *,
        scale: float,
        mask: object = None,
        sinks: object = None,
        **kwargs: Any,
    ) -> mx.array:
        call_config = (
            _effective_config(config, q.shape[1], k.shape[1])
            if q.ndim == 4 and k.ndim == 4
            else None
        )
        if call_config is not None:
            call_config = _request_config(call_config, k.shape[-2], total_tokens)
        if call_config is not None and _eligible(
            q, k, v, mask, sinks, kwargs, call_config, min_tokens
        ):
            return fpv2_sdpa(q, k, v, scale=scale, config=call_config)
        if _collect_stats and call_config is not None and _eligible(
            q, k, v, mask, sinks, kwargs, call_config, 1
        ):
            global _selected_dense_blocks, _visible_blocks
            visible = _causal_block_count(
                k.shape[1], q.shape[-2], k.shape[-2] - q.shape[-2], call_config
            )
            _selected_dense_blocks += visible
            _visible_blocks += visible
        return _ORIGINAL_SDPA(
            q, k, v, scale=scale, mask=mask, sinks=sinks, **kwargs
        )

    mx.fast.scaled_dot_product_attention = wrapped
    _enabled = True


def disable() -> None:
    """Restore MLX's original scaled dot-product attention function."""
    global _collect_stats, _enabled
    mx.fast.scaled_dot_product_attention = _ORIGINAL_SDPA
    _collect_stats = False
    _enabled = False


def is_enabled() -> bool:
    """Return whether the FPV2 SDPA hook is installed."""
    return _enabled
