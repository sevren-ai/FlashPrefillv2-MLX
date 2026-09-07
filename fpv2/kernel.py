"""Tiled block-sparse attention derived from MLX's Steel attention kernel.

Dense Steel attention sweeps contiguous KV tiles. This variant instead follows
an ascending block list, then appends pooled representatives of safely pruned
blocks to the same online softmax. Query tiles remain resident in threadgroup
memory while K and V alias a second buffer. Chunk offsets affect only absolute
causal positions, and per-dimension tile dispatch keeps the two matrix
multiplications, which dominate runtime, efficient. Row tiles are scheduled
latest-first because causal work grows with position, reducing the dispatch
tail. Optional PackGQA interleaves a KV head's query group to share K/V loads;
it is off by default because its transposes outweigh the bandwidth savings on
compute-bound M1 Pro workloads.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files

import mlx.core as mx

from fpv2.selection import FPV2Config


SUPPORTED_HEAD_DIMS = (64, 96, 128, 256)


@dataclass(frozen=True)
class TileConfig:
    """Compile-time tile dimensions for the Steel attention kernel."""

    bq: int
    bk: int
    wm: int

    @property
    def threads(self) -> int:
        return 32 * self.wm

    def check(self, head_dim: int, *, block_k: int = 128, tile_q: int = 128) -> None:
        if min(self.bq, self.bk, self.wm) <= 0:
            raise ValueError("tile dimensions must be positive")
        if self.bq % (8 * self.wm):
            raise ValueError("bq must contain whole 8-row fragments per simdgroup")
        if block_k % self.bk:
            raise ValueError("bk must divide block_k")
        if tile_q < self.bq or tile_q % self.bq:
            raise ValueError("bq must divide tile_q")
        if (self.bk * head_dim) % self.threads:
            raise ValueError("thread count must divide the correction gather")
        memory = (self.bq * (head_dim + 8) + max(
            head_dim * (self.bk + 8), self.bk * (head_dim + 8)
        )) * 2
        if memory > 32 * 1024:
            raise ValueError("tile exceeds 32 KB of threadgroup memory")


_DEFAULT_TILES = {
    64: TileConfig(32, 32, 4),
    96: TileConfig(32, 32, 4),
    128: TileConfig(64, 32, 8),
    256: TileConfig(32, 16, 4),
}


def default_tile(head_dim: int) -> TileConfig:
    """Return the measured default tile for a supported head dimension."""
    try:
        return _DEFAULT_TILES[head_dim]
    except KeyError as error:
        raise ValueError(f"head dimension must be one of {SUPPORTED_HEAD_DIMS}") from error


def _tile_for_config(head_dim: int, config: FPV2Config) -> TileConfig:
    tile = default_tile(head_dim)
    if (
        config.tile_q >= tile.bq
        and config.tile_q % tile.bq == 0
        and config.block_k % tile.bk == 0
    ):
        return tile
    if config.tile_q % 8 or config.block_k % 8:
        raise ValueError("block_k and tile_q must be multiples of 8 for Steel")
    bq = min(config.tile_q, 32)
    bk = min(config.block_k, 16)
    return TileConfig(bq, bk, bq // 8)

_STEEL_HEADERS = (
    "defines.h",
    "utils/type_traits.h",
    "utils/integral_constant.h",
    "utils.h",
    "attn/params.h",
    "attn/transforms.h",
    "attn/loader.h",
    "attn/mma.h",
)


@lru_cache(maxsize=1)
def assemble_steel_header() -> str:
    """Assemble vendored Steel headers for MLX's source-only compiler."""
    preamble = """#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;
#ifndef M_LOG2E_F
#define M_LOG2E_F 1.4426950408889634f
#endif
"""
    chunks = [preamble]
    root = files("fpv2").joinpath("metal", "steel")
    for relative_path in _STEEL_HEADERS:
        text = root.joinpath(*relative_path.split("/")).read_text()
        lines = [
            line
            for line in text.splitlines()
            if line.strip() != "#pragma once"
            and not (
                line.lstrip().startswith("#include") and '"mlx/' in line
            )
        ]
        chunks.append("\n".join(lines))
    chunks.append(
        r"""
using namespace mlx::steel;
struct MaxOp { static METAL_FUNC float apply(float x, float y) { return max(x, y); } };
struct SumOp { static METAL_FUNC float apply(float x, float y) { return x + y; } };
struct MulOp { static METAL_FUNC float apply(float x, float y) { return x * y; } };
struct ExpSubOp {
  static METAL_FUNC float apply(float x, float y) { return fast::exp2(x - y); }
};
struct DivOp { static METAL_FUNC float apply(float x, float y) { return x / y; } };
"""
    )
    return "\n".join(chunks)


def compact_block_indices(
    selected: mx.array,
    *,
    config: FPV2Config = FPV2Config(),
    offset: int = 0,
    query_length: int | None = None,
    key_length: int | None = None,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Create padded selected/pruned block lists and per-row counts lazily."""
    if selected.ndim != 3:
        raise ValueError("selected must have shape [KV heads, tiles, blocks]")
    if selected.dtype != mx.bool_:
        raise ValueError("selected must have boolean dtype")
    _, num_tiles, num_blocks = selected.shape
    padded_query_length = num_tiles * config.tile_q
    valid_query_length = (
        padded_query_length if query_length is None else query_length
    )
    represented_key_length = num_blocks * config.block_k
    valid_key_length = represented_key_length if key_length is None else key_length
    if offset < 0:
        raise ValueError("offset must be non-negative")
    if not 0 < valid_query_length <= padded_query_length:
        raise ValueError("query_length is inconsistent with selected tiles")
    if not 0 < valid_key_length <= represented_key_length:
        raise ValueError("key_length is inconsistent with selected blocks")
    if offset + valid_query_length > valid_key_length:
        raise ValueError("queries extend beyond key_length")

    selected_i32 = selected.astype(mx.int32)
    selected_ids = mx.argsort(-selected_i32, axis=-1).astype(mx.int32)
    selected_counts = mx.sum(selected_i32, axis=-1).astype(mx.int32)

    tile_first = offset + mx.arange(num_tiles, dtype=mx.int32) * config.tile_q
    block_ends = mx.minimum(
        (mx.arange(num_blocks, dtype=mx.int32) + 1) * config.block_k,
        valid_key_length,
    )
    fully_past = block_ends[None, :] <= tile_first[:, None]
    pruned = (~selected) & fully_past[None, :, :]
    pruned_i32 = pruned.astype(mx.int32)
    pruned_ids = mx.argsort(-pruned_i32, axis=-1).astype(mx.int32)
    pruned_counts = mx.sum(pruned_i32, axis=-1).astype(mx.int32)
    return selected_ids, selected_counts, pruned_ids, pruned_counts


_SOURCE = r"""
    constexpr int THREADS = 32 * WM;
    constexpr int LDQ = BD + 8;
    constexpr int LDK = BK + 8;
    constexpr int LDV = BD + 8;
    constexpr int TM = BQ / (8 * WM);
    constexpr int TNK = BK / 8;
    constexpr int TND = BD / 8;

    const int Q_ROWS = params[0];
    const int LK = params[1];
    const int NT = params[4];
    const int NB = params[5];
    const int OFFSET = params[6];
    const int BLOCK_K = params[7];
    const int TILE_Q = params[8];
    const int VALID_LK = params[9];
    const int VALID_LQ = params[10];
    const int G = params[11];
    const int Q_BLOCKS = params[12];

    const int q_block = Q_BLOCKS - 1 - int(threadgroup_position_in_grid.x);
    const int q_plane = int(threadgroup_position_in_grid.y);
    const int q_start = q_block * BQ;
    if (q_start >= Q_ROWS) return;
    const int kv_head = PACK_GQA ? q_plane : q_plane / G;
    const int first_position = PACK_GQA ? q_start / G : q_start;
    const int selection_tile = first_position / TILE_Q;
    const int list_row = (kv_head * NT + selection_tile) * NB;
    const int valid_q_rows = PACK_GQA ? VALID_LQ * G : VALID_LQ;
    const int rows_this_block = min(BQ, Q_ROWS - q_start);
    const int last_packed_row = min(q_start + BQ, valid_q_rows) - 1;
    const int last_position = PACK_GQA ? last_packed_row / G : last_packed_row;
    const int absolute_last = OFFSET + last_position;

    threadgroup T Qs[BQ * LDQ];
    threadgroup T KVs[(BD * LDK > BK * LDV) ? BD * LDK : BK * LDV];
    const device T* q_src = q + (q_plane * Q_ROWS + q_start) * BD;
    BlockLoader<T, BQ, BD, LDQ, 0, THREADS> q_loader(
        q_src, BD, Qs, simdgroup_index_in_threadgroup,
        thread_index_in_simdgroup);
    q_loader.load_safe(short2(BD, rows_this_block));
    threadgroup_barrier(mem_flags::mem_threadgroup);

    using QKMMA = BlockMMA<
        T, float, BQ, BK, BD, WM, 1, false, false, LDQ, LDK, float>;
    QKMMA qk_mma(simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
    using OutTile = MMATile<float, TM, TND>;
    using VTile = MMATile<float, TNK, TND>;
    OutTile O;
    O.clear();
    float running_max[OutTile::kRowsPerThread];
    float running_sum[OutTile::kRowsPerThread];
    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < OutTile::kRowsPerThread; ++i) {
        running_max[i] = Limits<float>::finite_min;
        running_sum[i] = 0.0f;
    }
    const short2 lane_coord = BaseMMAFrag<float, 8, 8>::get_coord(
        thread_index_in_simdgroup);

    const int selected_count = selected_counts[kv_head * NT + selection_tile];
    for (int list_idx = 0; list_idx < selected_count; ++list_idx) {
        const int block = selected_ids[list_row + list_idx];
        const int block_start = block * BLOCK_K;
        if (block_start > absolute_last) break;
        for (int sub = 0; sub < BLOCK_K; sub += BK) {
            const int key_start = block_start + sub;
            if (key_start > absolute_last) break;
            const device T* k_src = k + (kv_head * LK + key_start) * BD;
            BlockLoaderT<T, BK, BD, 1, LDK, 0, THREADS> k_loader(
                k_src, BD, KVs, simdgroup_index_in_threadgroup,
                thread_index_in_simdgroup);
            k_loader.load_safe(short2(BD, max(0, min(BK, VALID_LK - key_start))));
            threadgroup_barrier(mem_flags::mem_threadgroup);

            qk_mma.Ctile.clear();
            qk_mma.mma(Qs, KVs);
            const bool needs_causal =
                key_start + BK - 1 > OFFSET + first_position;
            STEEL_PRAGMA_UNROLL
            for (int i = 0; i < TM; ++i) {
                STEEL_PRAGMA_UNROLL
                for (int j = 0; j < TNK; ++j) {
                    thread auto& frag = qk_mma.Ctile.frag_at(i, j);
                    STEEL_PRAGMA_UNROLL
                    for (int e = 0; e < 2; ++e) {
                        const int local_row = qk_mma.sm + i * 8 * WM;
                        const int packed_row = q_start + local_row;
                        const int position = PACK_GQA ? packed_row / G : packed_row;
                        const int key_position = key_start + qk_mma.sn + j * 8 + e;
                        const bool valid = packed_row < valid_q_rows &&
                            key_position < VALID_LK &&
                            (!needs_causal || key_position <= OFFSET + position);
                        frag[e] = valid ? frag[e] * scales[0]
                                        : Limits<float>::finite_min;
                    }
                }
            }

            float tile_max[OutTile::kRowsPerThread];
            STEEL_PRAGMA_UNROLL
            for (int i = 0; i < OutTile::kRowsPerThread; ++i)
                tile_max[i] = Limits<float>::finite_min;
            qk_mma.Ctile.template row_reduce<MaxOp>(tile_max);
            float new_max[OutTile::kRowsPerThread];
            float rescale[OutTile::kRowsPerThread];
            STEEL_PRAGMA_UNROLL
            for (int i = 0; i < OutTile::kRowsPerThread; ++i) {
                new_max[i] = max(running_max[i], tile_max[i]);
                rescale[i] = fast::exp2(running_max[i] - new_max[i]);
                running_sum[i] *= rescale[i];
            }
            O.template row_bin_op<MulOp>(rescale);
            qk_mma.Ctile.template row_bin_op<ExpSubOp>(new_max);
            float tile_sum[OutTile::kRowsPerThread];
            STEEL_PRAGMA_UNROLL
            for (int i = 0; i < OutTile::kRowsPerThread; ++i) tile_sum[i] = 0.0f;
            qk_mma.Ctile.template row_reduce<SumOp>(tile_sum);
            STEEL_PRAGMA_UNROLL
            for (int i = 0; i < OutTile::kRowsPerThread; ++i) {
                running_sum[i] += tile_sum[i];
                running_max[i] = new_max[i];
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
            const device T* v_src = v + (kv_head * LK + key_start) * BD;
            BlockLoader<T, BK, BD, LDV, 0, THREADS> v_loader(
                v_src, BD, KVs, simdgroup_index_in_threadgroup,
                thread_index_in_simdgroup);
            v_loader.load_safe(short2(BD, max(0, min(BK, VALID_LK - key_start))));
            threadgroup_barrier(mem_flags::mem_threadgroup);
            VTile Vfrag;
            Vfrag.template load<T, 1, 1, LDV, 1>(
                KVs + lane_coord.y * LDV + lane_coord.x);
            tile_matmad(O, qk_mma.Ctile, Vfrag, O);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    if (DO_CORR) {
        const int pruned_count = pruned_counts[kv_head * NT + selection_tile];
        for (int list_start = 0; list_start < pruned_count; list_start += BK) {
            for (int linear = int(thread_index_in_threadgroup);
                 linear < BK * BD; linear += THREADS) {
                const int entry = linear / BD;
                const int dim = linear - entry * BD;
                const bool valid = list_start + entry < pruned_count;
                const int block = pruned_ids[list_row + (valid ? list_start + entry : 0)];
                KVs[dim * LDK + entry] = pooled_k[(kv_head * NB + block) * BD + dim];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            qk_mma.Ctile.clear();
            qk_mma.mma(Qs, KVs);
            STEEL_PRAGMA_UNROLL
            for (int i = 0; i < TM; ++i) {
                STEEL_PRAGMA_UNROLL
                for (int j = 0; j < TNK; ++j) {
                    thread auto& frag = qk_mma.Ctile.frag_at(i, j);
                    STEEL_PRAGMA_UNROLL
                    for (int e = 0; e < 2; ++e) {
                        const int entry = qk_mma.sn + j * 8 + e;
                        frag[e] = list_start + entry < pruned_count
                            ? frag[e] * scales[0] + scales[1]
                            : Limits<float>::finite_min;
                    }
                }
            }
            float tile_max[OutTile::kRowsPerThread];
            STEEL_PRAGMA_UNROLL
            for (int i = 0; i < OutTile::kRowsPerThread; ++i)
                tile_max[i] = Limits<float>::finite_min;
            qk_mma.Ctile.template row_reduce<MaxOp>(tile_max);
            float new_max[OutTile::kRowsPerThread];
            float rescale[OutTile::kRowsPerThread];
            STEEL_PRAGMA_UNROLL
            for (int i = 0; i < OutTile::kRowsPerThread; ++i) {
                new_max[i] = max(running_max[i], tile_max[i]);
                rescale[i] = fast::exp2(running_max[i] - new_max[i]);
                running_sum[i] *= rescale[i];
            }
            O.template row_bin_op<MulOp>(rescale);
            qk_mma.Ctile.template row_bin_op<ExpSubOp>(new_max);
            float tile_sum[OutTile::kRowsPerThread];
            STEEL_PRAGMA_UNROLL
            for (int i = 0; i < OutTile::kRowsPerThread; ++i) tile_sum[i] = 0.0f;
            qk_mma.Ctile.template row_reduce<SumOp>(tile_sum);
            STEEL_PRAGMA_UNROLL
            for (int i = 0; i < OutTile::kRowsPerThread; ++i) {
                running_sum[i] += tile_sum[i];
                running_max[i] = new_max[i];
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int linear = int(thread_index_in_threadgroup);
                 linear < BK * BD; linear += THREADS) {
                const int entry = linear / BD;
                const int dim = linear - entry * BD;
                const bool valid = list_start + entry < pruned_count;
                const int block = pruned_ids[list_row + (valid ? list_start + entry : 0)];
                KVs[entry * LDV + dim] = pooled_v[(kv_head * NB + block) * BD + dim];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            VTile Vfrag;
            Vfrag.template load<T, 1, 1, LDV, 1>(
                KVs + lane_coord.y * LDV + lane_coord.x);
            tile_matmad(O, qk_mma.Ctile, Vfrag, O);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    O.template row_bin_op<DivOp>(running_sum);
    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < TM; ++i) {
        STEEL_PRAGMA_UNROLL
        for (int j = 0; j < TND; ++j) {
            thread const auto& frag = O.frag_at(i, j);
            STEEL_PRAGMA_UNROLL
            for (int e = 0; e < 2; ++e) {
                const int local_row = qk_mma.sm + i * 8 * WM;
                const int col = lane_coord.x + j * 8 + e;
                if (local_row < rows_this_block && col < BD) {
                    out[(q_plane * Q_ROWS + q_start + local_row) * BD + col] = T(frag[e]);
                }
            }
        }
    }
"""


@lru_cache(maxsize=1)
def _attention_kernel():
    return mx.fast.metal_kernel(
        name="fpv2_sparse_attention",
        input_names=[
            "q",
            "k",
            "v",
            "pooled_k",
            "pooled_v",
            "selected_ids",
            "selected_counts",
            "pruned_ids",
            "pruned_counts",
            "params",
            "scales",
        ],
        output_names=["out"],
        source=_SOURCE,
        header=assemble_steel_header(),
        compile_options={"math_mode": "fast"},
    )


def sparse_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    pooled_k: mx.array,
    pooled_v: mx.array,
    selected_ids: mx.array,
    selected_counts: mx.array,
    pruned_ids: mx.array,
    pruned_counts: mx.array,
    *,
    scale: float,
    config: FPV2Config = FPV2Config(),
    offset: int = 0,
    query_length: int | None = None,
    key_length: int | None = None,
    tile: TileConfig | None = None,
    pack_gqa: bool = False,
) -> mx.array:
    """Run tiled sparse attention, optionally sharing K/V loads with PackGQA.

    PackGQA is disabled by default because this kernel is matmul-bound on M1
    Pro; query packing and output unpacking cost more than the saved KV traffic.
    """
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("q, k, and v must have shape [heads, sequence, dim]")
    if k.shape != v.shape:
        raise ValueError("k and v must have identical shapes")
    if q.dtype not in (mx.float16, mx.bfloat16) or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("q, k, and v must share float16 or bfloat16 dtype")
    query_heads, padded_query_length, head_dim = q.shape
    kv_heads, padded_key_length, key_dim = k.shape
    if head_dim != key_dim or head_dim not in SUPPORTED_HEAD_DIMS:
        raise ValueError(f"head dimension must be one of {SUPPORTED_HEAD_DIMS}")
    if not kv_heads or query_heads % kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    if padded_query_length % config.tile_q or padded_key_length % config.block_k:
        raise ValueError("q and k/v sequence lengths must be tile/block aligned")
    valid_query_length = (
        padded_query_length if query_length is None else query_length
    )
    valid_key_length = padded_key_length if key_length is None else key_length
    if not 0 < valid_query_length <= padded_query_length:
        raise ValueError("query_length must fit within padded q")
    if not 0 < valid_key_length <= padded_key_length:
        raise ValueError("key_length must fit within padded k/v")
    num_tiles = padded_query_length // config.tile_q
    num_blocks = padded_key_length // config.block_k
    expected_lists = (kv_heads, num_tiles, num_blocks)
    expected_counts = (kv_heads, num_tiles)
    if selected_ids.shape != expected_lists or pruned_ids.shape != expected_lists:
        raise ValueError(f"block lists must have shape {expected_lists}")
    if selected_counts.shape != expected_counts or pruned_counts.shape != expected_counts:
        raise ValueError(f"block counts must have shape {expected_counts}")
    lists = (selected_ids, pruned_ids)
    counts = (selected_counts, pruned_counts)
    if any(array.dtype != mx.int32 for array in (*lists, *counts)):
        raise ValueError("block lists and counts must have int32 dtype")
    if pooled_k.shape != (kv_heads, num_blocks, head_dim) or pooled_v.shape != pooled_k.shape:
        raise ValueError("pooled k/v shapes do not match padded k/v")
    if offset < 0 or offset + valid_query_length > valid_key_length:
        raise ValueError("offset does not align queries within the key sequence")

    tile = _tile_for_config(head_dim, config) if tile is None else tile
    tile.check(head_dim, block_k=config.block_k, tile_q=config.tile_q)
    group_size = query_heads // kv_heads
    if pack_gqa and (
        tile.bq % group_size or (config.tile_q * group_size) % tile.bq
    ):
        raise ValueError("PackGQA tile must align with complete GQA groups")

    kernel_q = q
    if pack_gqa:
        kernel_q = mx.reshape(
            mx.transpose(
                mx.reshape(
                    q,
                    (kv_heads, group_size, padded_query_length, head_dim),
                ),
                (0, 2, 1, 3),
            ),
            (kv_heads, padded_query_length * group_size, head_dim),
        )
    kernel_query_rows = kernel_q.shape[1]
    kernel_planes = kernel_q.shape[0]
    query_blocks = (kernel_query_rows + tile.bq - 1) // tile.bq

    params = mx.array(
        [
            kernel_query_rows,
            padded_key_length,
            query_heads,
            kv_heads,
            num_tiles,
            num_blocks,
            offset,
            config.block_k,
            config.tile_q,
            valid_key_length,
            valid_query_length,
            group_size,
            query_blocks,
        ],
        dtype=mx.int32,
    )
    scales = mx.array(
        [scale * 1.4426950408889634, math.log2(config.block_k)], dtype=mx.float32
    )
    inputs = [
        kernel_q,
        k,
        v,
        pooled_k.astype(q.dtype),
        pooled_v.astype(q.dtype),
        mx.clip(selected_ids, 0, num_blocks - 1),
        mx.clip(selected_counts, 0, num_blocks),
        mx.clip(pruned_ids, 0, num_blocks - 1),
        mx.clip(pruned_counts, 0, num_blocks),
        params,
        scales,
    ]
    output = _attention_kernel()(
        inputs=inputs,
        output_shapes=[kernel_q.shape],
        output_dtypes=[q.dtype],
        grid=(query_blocks * 32, kernel_planes * tile.wm, 1),
        threadgroup=(32, tile.wm, 1),
        template=[
            ("T", q.dtype),
            ("DO_CORR", config.mean_correction),
            ("PACK_GQA", pack_gqa),
            ("BQ", tile.bq),
            ("BK", tile.bk),
            ("BD", head_dim),
            ("WM", tile.wm),
            ("WN", 1),
        ],
    )[0]
    if not pack_gqa:
        return output
    return mx.reshape(
        mx.transpose(
            mx.reshape(
                output,
                (kv_heads, padded_query_length, group_size, head_dim),
            ),
            (0, 2, 1, 3),
        ),
        q.shape,
    )
