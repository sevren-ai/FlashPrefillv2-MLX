from __future__ import annotations

from dataclasses import replace

import mlx.core as mx
import pytest

from fpv2.selection import FPV2Config, block_scores_and_select, density, pool_blocks
from tests.synthetic import structured_qkv


def test_pool_blocks() -> None:
    x = mx.reshape(mx.arange(32, dtype=mx.float32), (2, 8, 2))
    pooled = pool_blocks(x, 4)
    expected = mx.mean(mx.reshape(x, (2, 2, 4, 2)), axis=2)
    assert mx.array_equal(pooled, expected).item()


def test_alpha_zero_selects_exactly_visible_blocks() -> None:
    config = FPV2Config(
        block_k=32, tile_q=32, alpha=0, sink_tokens=0, window_tokens=0
    )
    q, k, _ = structured_qkv(
        length=256, query_heads=8, kv_heads=2, block_size=config.block_k
    )
    selected, _ = block_scores_and_select(
        q, pool_blocks(k, config.block_k), scale=q.shape[-1] ** -0.5, config=config
    )
    expected = mx.arange(8)[None, None, :] <= mx.arange(8)[None, :, None]
    expected = mx.broadcast_to(expected, selected.shape)
    assert mx.array_equal(selected, expected).item()
    assert mx.allclose(density(selected, config=config), mx.array(1.0)).item()


def test_pool_blocks_rejects_unaligned_length() -> None:
    with pytest.raises(ValueError, match="divisible"):
        pool_blocks(mx.zeros((2, 7, 4)), 4)


def test_partially_visible_block_does_not_set_tile_energy_scale() -> None:
    config = FPV2Config(
        block_k=4,
        tile_q=2,
        alpha=0.5,
        sink_tokens=0,
        window_tokens=0,
        last_n_blocks=0,
    )
    q = mx.ones((1, 2, 1))
    pooled_k = mx.array([[[0.0], [10.0]]])
    selected, _ = block_scores_and_select(
        q, pooled_k, scale=1.0, config=config, offset=4
    )
    mx.eval(selected)

    # Block 0 wins the scored energy; block 1 is retained only as the diagonal.
    assert selected.tolist() == [[[True, True]]]


def test_last_n_blocks_are_fully_dense_for_complete_request() -> None:
    config = FPV2Config(
        block_k=4,
        tile_q=4,
        alpha=1e9,
        sink_tokens=0,
        window_tokens=0,
        last_n_blocks=2,
    )
    q = mx.ones((1, 16, 1))
    pooled_k = mx.ones((1, 4, 1))
    selected, _ = block_scores_and_select(q, pooled_k, scale=1.0, config=config)
    mx.eval(selected)

    assert selected.tolist() == [
        [
            [True, False, False, False],
            [False, True, False, False],
            [True, True, True, False],
            [True, True, True, True],
        ]
    ]


def test_hopper_selection_defaults() -> None:
    config = FPV2Config()
    assert config.window_tokens == 384
    assert config.last_n_blocks == 2


def test_chunked_scoring_matches_unchunked_selection() -> None:
    config = FPV2Config(
        block_k=16, tile_q=16, alpha=0.35, sink_tokens=16, window_tokens=32
    )
    q, k, _ = structured_qkv(
        length=160, query_heads=8, kv_heads=2, block_size=config.block_k, seed=7
    )
    pooled_k = pool_blocks(k, config.block_k)
    expected, scores = block_scores_and_select(
        q, pooled_k, scale=q.shape[-1] ** -0.5, config=config
    )
    actual, chunked_scores = block_scores_and_select(
        q,
        pooled_k,
        scale=q.shape[-1] ** -0.5,
        config=replace(config, score_chunk_tiles=3),
    )
    mx.eval(expected, actual)

    assert scores is not None
    assert chunked_scores is None
    assert mx.array_equal(actual, expected).item()


def test_chunked_scoring_preserves_offset_and_padded_query_semantics() -> None:
    config = FPV2Config(
        block_k=16, tile_q=16, alpha=0.4, sink_tokens=16, window_tokens=16
    )
    q, k, _ = structured_qkv(
        length=160, query_heads=4, kv_heads=2, block_size=config.block_k, seed=11
    )
    offset = 37
    query_length = 55
    padded_q = mx.pad(q[:, offset : offset + query_length], [(0, 0), (0, 9), (0, 0)])
    pooled_k = pool_blocks(k, config.block_k)
    expected, _ = block_scores_and_select(
        padded_q,
        pooled_k,
        scale=q.shape[-1] ** -0.5,
        config=config,
        offset=offset,
        query_length=query_length,
    )
    actual, scores = block_scores_and_select(
        padded_q,
        pooled_k,
        scale=q.shape[-1] ** -0.5,
        config=replace(config, score_chunk_tiles=2),
        offset=offset,
        query_length=query_length,
    )
    mx.eval(expected, actual)

    assert scores is None
    assert mx.array_equal(actual, expected).item()


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_score_chunk_tiles_must_be_a_positive_integer(value: object) -> None:
    with pytest.raises(ValueError, match="score_chunk_tiles"):
        FPV2Config(score_chunk_tiles=value)  # type: ignore[arg-type]


def test_pack_gqa_must_be_boolean() -> None:
    with pytest.raises(ValueError, match="pack_gqa"):
        FPV2Config(pack_gqa=1)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [-1, 1.5, True])
def test_last_n_blocks_must_be_a_non_negative_integer(value: object) -> None:
    with pytest.raises(ValueError, match="last_n_blocks"):
        FPV2Config(last_n_blocks=value)  # type: ignore[arg-type]
