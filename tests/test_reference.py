from __future__ import annotations

import mlx.core as mx

from fpv2.reference import fpv2_attention_reference
from fpv2.selection import FPV2Config, block_scores_and_select, density, pool_blocks
from tests.synthetic import structured_qkv


def exact_causal_attention(
    q: mx.array, k: mx.array, v: mx.array, scale: float, offset: int = 0
) -> mx.array:
    hq, query_length, dim = q.shape
    key_length = k.shape[1]
    hkv = k.shape[0]
    grouped_q = mx.reshape(
        q.astype(mx.float32), (hkv, hq // hkv, query_length, dim)
    )
    scores = mx.einsum("hgnd,hmd->hgnm", grouped_q, k.astype(mx.float32)) * scale
    causal = mx.arange(key_length)[None, :] <= (
        offset + mx.arange(query_length)[:, None]
    )
    scores = mx.where(causal[None, None, :, :], scores, -mx.inf)
    weights = mx.softmax(scores, axis=-1)
    output = mx.einsum("hgnm,hmd->hgnd", weights, v.astype(mx.float32))
    return mx.reshape(output, (hq, query_length, dim))


def test_dense_selection_matches_exact_causal_attention() -> None:
    config = FPV2Config(
        block_k=32,
        tile_q=32,
        alpha=0,
        sink_tokens=0,
        window_tokens=0,
        mean_correction=False,
    )
    q, k, v = structured_qkv(length=256, block_size=config.block_k)
    scale = q.shape[-1] ** -0.5
    actual = fpv2_attention_reference(q, k, v, scale=scale, config=config)
    expected = exact_causal_attention(q, k, v, scale)
    mx.eval(actual, expected)
    relative_error = mx.max(mx.abs(actual - expected)) / mx.maximum(
        mx.max(mx.abs(expected)), 1e-8
    )
    assert relative_error.item() < 1e-5

    selected, _ = block_scores_and_select(
        q, pool_blocks(k, config.block_k), scale=scale, config=config
    )
    assert density(selected, config=config).item() == 1.0


def test_mean_correction_improves_sparse_approximation() -> None:
    base = dict(block_k=32, tile_q=32, alpha=0.6, sink_tokens=32, window_tokens=32)
    q, k, v = structured_qkv(length=512, block_size=32, noise=0.01)
    scale = 3.0
    expected = exact_causal_attention(q, k, v, scale)
    truncated = fpv2_attention_reference(
        q, k, v, scale=scale, config=FPV2Config(**base, mean_correction=False)
    )
    corrected = fpv2_attention_reference(
        q, k, v, scale=scale, config=FPV2Config(**base, mean_correction=True)
    )
    mx.eval(expected, truncated, corrected)
    truncated_error = mx.max(mx.abs(truncated - expected)).item()
    corrected_error = mx.max(mx.abs(corrected - expected)).item()
    assert corrected_error <= truncated_error


def test_dense_selection_matches_exact_attention_with_offset() -> None:
    config = FPV2Config(
        block_k=128,
        tile_q=128,
        alpha=0,
        sink_tokens=0,
        window_tokens=0,
        mean_correction=False,
    )
    q, k, v = structured_qkv(length=2_048, block_size=config.block_k, seed=13)
    offset = 1_024
    q_chunk = q[:, offset:]
    scale = q.shape[-1] ** -0.5
    actual = fpv2_attention_reference(
        q_chunk, k, v, scale=scale, config=config, offset=offset
    )
    expected = exact_causal_attention(q_chunk, k, v, scale, offset)
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, rtol=1e-5, atol=1e-6).item()
