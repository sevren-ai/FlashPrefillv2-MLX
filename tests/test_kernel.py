from __future__ import annotations

import mlx.core as mx
import pytest

from fpv2.kernel import (
    TileConfig,
    assemble_steel_header,
    compact_block_indices,
    default_tile,
    sparse_attention,
)
from fpv2.reference import fpv2_attention_reference
from fpv2.selection import FPV2Config, block_scores_and_select, pool_blocks
from tests.synthetic import structured_qkv


@pytest.mark.parametrize("mean_correction", [False, True])
@pytest.mark.parametrize("head_dim", [64, 96, 128, 256])
@pytest.mark.parametrize("pack_gqa", [False, True])
def test_kernel_matches_reference(
    mean_correction: bool, head_dim: int, pack_gqa: bool
) -> None:
    config = FPV2Config(
        block_k=16,
        tile_q=16,
        alpha=0.15,
        sink_tokens=16,
        window_tokens=16,
        mean_correction=mean_correction,
    )
    q, k, v = structured_qkv(
        length=64,
        query_heads=4,
        kv_heads=2,
        head_dim=head_dim,
        block_size=config.block_k,
        dtype=mx.bfloat16,
    )
    scale = q.shape[-1] ** -0.5
    pooled_k = pool_blocks(k, config.block_k)
    pooled_v = pool_blocks(v, config.block_k)
    selected, _ = block_scores_and_select(
        q, pooled_k, scale=scale, config=config
    )
    indices = compact_block_indices(selected, config=config)
    actual = sparse_attention(
        q,
        k,
        v,
        pooled_k,
        pooled_v,
        *indices,
        scale=scale,
        config=config,
        pack_gqa=pack_gqa,
    )
    expected = fpv2_attention_reference(q, k, v, scale=scale, config=config)
    mx.eval(actual, expected)
    relative_error = mx.max(mx.abs(actual.astype(mx.float32) - expected)) / mx.maximum(
        mx.max(mx.abs(expected)), 1e-8
    )
    assert relative_error.item() < 2e-2


def test_compaction_preserves_ascending_selected_indices() -> None:
    selected = mx.array([[[True, False, True, False]]])
    config = FPV2Config(block_k=16, tile_q=16)
    ids, counts, _, _ = compact_block_indices(selected, config=config)
    mx.eval(ids, counts)
    assert counts.item() == 2
    assert ids[0, 0, :2].tolist() == [0, 2]


def test_compaction_rejects_non_boolean_mask() -> None:
    with pytest.raises(ValueError, match="boolean"):
        compact_block_indices(mx.ones((1, 1, 1), dtype=mx.int32))


def test_steel_header_assembly_removes_local_includes() -> None:
    header = assemble_steel_header()
    assert "#pragma once" not in header
    assert '#include "mlx/' not in header
    assert "namespace mlx" in header
    assert "namespace steel" in header
    assert "struct ExpSubOp" in header


def assert_kernel_matches_reference(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    config: FPV2Config,
    *,
    offset: int = 0,
    pack_gqa: bool = False,
    tile: TileConfig | None = None,
) -> None:
    scale = q.shape[-1] ** -0.5
    pooled_k = pool_blocks(k, config.block_k)
    pooled_v = pool_blocks(v, config.block_k)
    selected, _ = block_scores_and_select(
        q, pooled_k, scale=scale, config=config, offset=offset
    )
    indices = compact_block_indices(
        selected,
        config=config,
        offset=offset,
        query_length=q.shape[1],
        key_length=k.shape[1],
    )
    actual = sparse_attention(
        q,
        k,
        v,
        pooled_k,
        pooled_v,
        *indices,
        scale=scale,
        config=config,
        offset=offset,
        pack_gqa=pack_gqa,
        tile=tile,
    )
    expected = fpv2_attention_reference(
        q, k, v, scale=scale, config=config, offset=offset
    )
    mx.eval(actual, expected)
    relative_error = mx.max(mx.abs(actual.astype(mx.float32) - expected)) / mx.maximum(
        mx.max(mx.abs(expected)), 1e-8
    )
    assert relative_error.item() < 2e-2


@pytest.mark.parametrize("pack_gqa", [False, True])
@pytest.mark.parametrize(
    ("alpha", "mean_correction"),
    [(0.0, False), (0.15, False), (0.15, True)],
)
def test_real_block_kernel_modes(alpha: float, mean_correction: bool, pack_gqa: bool) -> None:
    config = FPV2Config(
        alpha=alpha,
        sink_tokens=128,
        window_tokens=256,
        mean_correction=mean_correction,
    )
    q, k, v = structured_qkv(
        length=2_048,
        query_heads=8,
        kv_heads=2,
        head_dim=128,
        dtype=mx.bfloat16,
        seed=31,
    )
    assert_kernel_matches_reference(q, k, v, config, pack_gqa=pack_gqa)


@pytest.mark.parametrize("pack_gqa", [False, True])
@pytest.mark.parametrize(("offset", "query_length"), [(1_024, 1_024), (1_920, 128)])
def test_chunk_offset_kernel(offset: int, query_length: int, pack_gqa: bool) -> None:
    config = FPV2Config(
        alpha=0.15,
        sink_tokens=128,
        window_tokens=256,
        mean_correction=True,
    )
    full_q, k, v = structured_qkv(
        length=2_048,
        query_heads=8,
        kv_heads=2,
        head_dim=128,
        dtype=mx.bfloat16,
        seed=37,
    )
    q = full_q[:, offset : offset + query_length]
    assert_kernel_matches_reference(
        q, k, v, config, offset=offset, pack_gqa=pack_gqa
    )


@pytest.mark.parametrize("head_dim", [64, 96, 256])
def test_real_block_supported_head_dims(head_dim: int) -> None:
    config = FPV2Config(
        alpha=0.15,
        sink_tokens=128,
        window_tokens=256,
        mean_correction=True,
    )
    q, k, v = structured_qkv(
        length=2_048,
        query_heads=8,
        kv_heads=2,
        head_dim=head_dim,
        dtype=mx.bfloat16,
        seed=41,
    )
    assert_kernel_matches_reference(q, k, v, config)


def test_default_tile_dispatch() -> None:
    assert default_tile(64) == TileConfig(32, 32, 4)
    assert default_tile(96) == TileConfig(32, 32, 4)
    assert default_tile(128) == TileConfig(64, 32, 8)
    assert default_tile(256) == TileConfig(32, 16, 4)


def test_multi_fragment_rows_match_reference() -> None:
    config = FPV2Config(alpha=0.15, sink_tokens=128, window_tokens=256)
    q, k, v = structured_qkv(
        length=2_048,
        query_heads=8,
        kv_heads=2,
        head_dim=128,
        dtype=mx.bfloat16,
        seed=43,
    )
    assert_kernel_matches_reference(
        q, k, v, config, tile=TileConfig(64, 32, 4)
    )
