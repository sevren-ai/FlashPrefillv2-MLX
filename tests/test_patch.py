from __future__ import annotations

import mlx.core as mx
import pytest

from fpv2 import FPV2Config, block_sparsity, disable, enable, fpv2_sdpa
from fpv2.patch import _ORIGINAL_SDPA, _effective_config, _request_config
from tests.synthetic import structured_qkv


@pytest.fixture(autouse=True)
def restore_sdpa():
    disable()
    yield
    disable()


def inputs(length: int, query_length: int | None = None):
    mx.random.seed(7)
    query_length = length if query_length is None else query_length
    q = mx.random.normal((1, 4, query_length, 64)).astype(mx.bfloat16)
    k = mx.random.normal((1, 2, length, 64)).astype(mx.bfloat16)
    v = mx.random.normal((1, 2, length, 64)).astype(mx.bfloat16)
    return q, k, v


def test_default_tile_uses_128_packed_gqa_rows() -> None:
    config = FPV2Config()
    effective = _effective_config(config, query_heads=32, kv_heads=8)
    assert effective is not None
    assert effective.tile_q == 32

    custom = FPV2Config(tile_q=64)
    assert _effective_config(custom, query_heads=32, kv_heads=8) is custom


def test_dense_tail_is_applied_only_at_known_request_end() -> None:
    config = FPV2Config(last_n_blocks=2)
    assert _request_config(config, 2_048, 8_192).last_n_blocks == 0
    assert _request_config(config, 8_192, 8_192) is config
    assert _request_config(config, 8_192, None).last_n_blocks == 0


def test_alpha_zero_padded_call_matches_stock() -> None:
    q, k, v = inputs(130)
    scale = q.shape[-1] ** -0.5
    expected = _ORIGINAL_SDPA(q, k, v, scale=scale, mask="causal")
    enable(
        FPV2Config(alpha=0, sink_tokens=0, window_tokens=0), min_tokens=1
    )
    actual = mx.fast.scaled_dot_product_attention(
        q, k, v, scale=scale, mask="causal"
    )
    mx.eval(actual, expected)
    error = mx.max(mx.abs(actual.astype(mx.float32) - expected.astype(mx.float32)))
    relative_error = error / mx.maximum(mx.max(mx.abs(expected)), 1e-8)
    assert relative_error.item() < 2e-2


def test_alpha_zero_unaligned_chunk_matches_stock() -> None:
    q, k, v = inputs(190, 128)
    scale = q.shape[-1] ** -0.5
    expected = _ORIGINAL_SDPA(q, k, v, scale=scale, mask="causal")
    enable(
        FPV2Config(alpha=0, sink_tokens=0, window_tokens=0), min_tokens=1
    )
    actual = mx.fast.scaled_dot_product_attention(
        q, k, v, scale=scale, mask="causal"
    )
    mx.eval(actual, expected)
    error = mx.max(mx.abs(actual.astype(mx.float32) - expected.astype(mx.float32)))
    relative_error = error / mx.maximum(mx.max(mx.abs(expected)), 1e-8)
    assert relative_error.item() < 2e-2


@pytest.mark.parametrize(
    ("length", "query_length", "mask"),
    [
        (64, 64, "causal"),
        (128, 1, "causal"),
        (128, 32, "causal"),
        (128, 128, None),
    ],
)
def test_ineligible_calls_are_bit_identical(
    length: int, query_length: int, mask: object
) -> None:
    q, k, v = inputs(length, query_length)
    scale = q.shape[-1] ** -0.5
    expected = _ORIGINAL_SDPA(q, k, v, scale=scale, mask=mask)
    enable(min_tokens=128)
    actual = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
    mx.eval(actual, expected)
    assert mx.array_equal(actual, expected).item()


def test_array_mask_falls_through_and_disable_restores_original() -> None:
    q, k, v = inputs(128)
    mask = mx.zeros((128, 128), dtype=mx.bfloat16)
    scale = q.shape[-1] ** -0.5
    expected = _ORIGINAL_SDPA(q, k, v, scale=scale, mask=mask)
    enable(min_tokens=1)
    actual = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
    mx.eval(actual, expected)
    assert mx.array_equal(actual, expected).item()
    disable()
    assert mx.fast.scaled_dot_product_attention is _ORIGINAL_SDPA


@pytest.mark.parametrize(
    ("key_length", "query_length"),
    [(4_096, 4_096), (4_100, 4_100), (8_192, 2_048), (4_100, 1_000), (6_000, 128)],
)
def test_real_shape_alpha_zero_matches_stock(
    key_length: int, query_length: int
) -> None:
    mx.random.seed(17)
    q = mx.random.normal((1, 8, query_length, 128)).astype(mx.bfloat16)
    k = mx.random.normal((1, 2, key_length, 128)).astype(mx.bfloat16)
    v = mx.random.normal(k.shape).astype(mx.bfloat16)
    scale = 128**-0.5
    expected = _ORIGINAL_SDPA(q, k, v, scale=scale, mask="causal")
    enable(
        FPV2Config(alpha=0, sink_tokens=0, window_tokens=0), min_tokens=1
    )
    actual = mx.fast.scaled_dot_product_attention(
        q, k, v, scale=scale, mask="causal"
    )
    mx.eval(actual, expected)
    relative_error = mx.max(
        mx.abs(actual.astype(mx.float32) - expected.astype(mx.float32))
    ) / mx.maximum(mx.max(mx.abs(expected)), 1e-8)
    assert relative_error.item() < 2e-2


def test_tile_aligned_chunks_match_single_shot_fpv2() -> None:
    q, k, v = structured_qkv(
        length=8_192,
        query_heads=8,
        kv_heads=2,
        head_dim=128,
        dtype=mx.bfloat16,
        seed=23,
    )
    q, k, v = q[None], k[None], v[None]
    config = FPV2Config(alpha=0.15, mean_correction=True)
    scale = 128**-0.5
    expected = fpv2_sdpa(q, k, v, scale=scale, config=config)
    chunks = []
    for end in range(2_048, 8_193, 2_048):
        chunk_config = _request_config(config, end, q.shape[-2])
        chunks.append(
            fpv2_sdpa(
                q[:, :, end - 2_048 : end],
                k[:, :, :end],
                v[:, :, :end],
                scale=scale,
                config=chunk_config,
            )
        )
    actual = mx.concatenate(chunks, axis=2)
    mx.eval(actual, expected)
    relative_error = mx.max(
        mx.abs(actual.astype(mx.float32) - expected.astype(mx.float32))
    ) / mx.maximum(mx.max(mx.abs(expected)), 1e-8)
    assert relative_error.item() < 1e-2


def test_large_array_mask_falls_through() -> None:
    q, k, v = inputs(4_096)
    mask = mx.zeros((4_096, 4_096), dtype=mx.bfloat16)
    scale = q.shape[-1] ** -0.5
    expected = _ORIGINAL_SDPA(q, k, v, scale=scale, mask=mask)
    enable(min_tokens=1)
    actual = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
    mx.eval(actual, expected)
    assert mx.array_equal(actual, expected).item()


def test_block_sparsity_includes_dense_gated_prefill() -> None:
    q, k, v = inputs(256)
    scale = q.shape[-1] ** -0.5
    enable(
        FPV2Config(
            alpha=1e9, sink_tokens=0, window_tokens=0, last_n_blocks=0
        ),
        min_tokens=256,
        collect_stats=True,
    )
    output = mx.fast.scaled_dot_product_attention(
        q, k, v, scale=scale, mask="causal"
    )
    mx.eval(output)
    assert block_sparsity() == pytest.approx(1 / 3)

    q, k, v = inputs(128)
    output = mx.fast.scaled_dot_product_attention(
        q, k, v, scale=scale, mask="causal"
    )
    mx.eval(output)
    assert block_sparsity() == pytest.approx(1 / 4)


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_total_tokens_must_be_a_positive_integer(value: object) -> None:
    with pytest.raises(ValueError, match="total_tokens"):
        enable(total_tokens=value)  # type: ignore[arg-type]
