"""Dense float32 reference implementation of FlashPrefill V2 attention."""

from __future__ import annotations

import mlx.core as mx

from fpv2.selection import FPV2Config, block_scores_and_select, pool_blocks


def fpv2_attention_reference(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    *,
    scale: float,
    config: FPV2Config = FPV2Config(),
    offset: int = 0,
) -> mx.array:
    """Compute the FPV2 approximation densely in float32.

    Inputs have shape ``[Hq, Lq, D]`` and ``[Hkv, Lk, D]``. Sequence lengths
    must already be padded to ``tile_q`` and ``block_k`` respectively.
    """
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("q, k, and v must have shape [heads, sequence, dim]")
    if k.shape != v.shape:
        raise ValueError("k and v must have identical shapes")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("q, k, and v head dimensions must match")
    if q.shape[0] % k.shape[0]:
        raise ValueError("query heads must be divisible by KV heads")
    if q.shape[1] % config.tile_q or k.shape[1] % config.block_k:
        raise ValueError("q and k/v sequence lengths must be tile/block aligned")
    if offset < 0 or offset + q.shape[1] > k.shape[1]:
        raise ValueError("offset does not align queries within the key sequence")

    query_heads, query_length, head_dim = q.shape
    kv_heads, key_length, _ = k.shape
    group_size = query_heads // kv_heads
    num_tiles = query_length // config.tile_q
    num_blocks = key_length // config.block_k
    grouped_q = mx.reshape(
        q.astype(mx.float32), (kv_heads, group_size, query_length, head_dim)
    )
    k32 = k.astype(mx.float32)
    v32 = v.astype(mx.float32)
    pooled_k = pool_blocks(k32, config.block_k)
    pooled_v = pool_blocks(v32, config.block_k)
    selected, pooled_scores = block_scores_and_select(
        q, pooled_k, scale=scale, config=config, offset=offset
    )
    if pooled_scores is None:
        pooled_scores = mx.einsum("hgnd,hbd->hgnb", grouped_q, pooled_k) * scale

    token_scores = mx.einsum("hgnd,hmd->hgnm", grouped_q, k32) * scale
    query_positions = offset + mx.arange(query_length, dtype=mx.int32)
    key_positions = mx.arange(key_length, dtype=mx.int32)
    causal = key_positions[None, :] <= query_positions[:, None]
    selected_tokens = mx.repeat(
        mx.repeat(selected, config.tile_q, axis=1), config.block_k, axis=2
    )[:, :query_length, :key_length]
    kept = selected_tokens[:, None, :, :] & causal[None, None, :, :]
    exact_logits = mx.where(kept, token_scores, -mx.inf)

    if config.mean_correction:
        tile_first = offset + mx.arange(num_tiles, dtype=mx.int32) * config.tile_q
        block_ends = (
            mx.arange(num_blocks, dtype=mx.int32) + 1
        ) * config.block_k
        fully_past = block_ends[None, :] <= tile_first[:, None]
        corrected = (~selected) & fully_past[None, :, :]
        corrected_rows = mx.repeat(corrected, config.tile_q, axis=1)
        correction_logits = pooled_scores + mx.log(float(config.block_k))
        correction_logits = mx.where(
            corrected_rows[:, None, :, :], correction_logits, -mx.inf
        )
    else:
        correction_logits = mx.full_like(pooled_scores, -mx.inf)

    row_max = mx.maximum(
        mx.max(exact_logits, axis=-1, keepdims=True),
        mx.max(correction_logits, axis=-1, keepdims=True),
    )
    exact_weights = mx.exp(exact_logits - row_max)
    correction_weights = mx.exp(correction_logits - row_max)
    numerator = mx.einsum("hgnm,hmd->hgnd", exact_weights, v32)
    numerator += mx.einsum("hgnb,hbd->hgnd", correction_weights, pooled_v)
    denominator = mx.sum(exact_weights, axis=-1, keepdims=True)
    denominator += mx.sum(correction_weights, axis=-1, keepdims=True)
    return mx.reshape(numerator / denominator, (query_heads, query_length, head_dim))
