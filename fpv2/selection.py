"""Pure-MLX block scoring and selection for FlashPrefill V2.

Keys are mean-pooled into blocks, ``k_bar[b] = mean(k[b])``. For every query
row and pooled key, the index stage computes ``s[q, b] = scale * q @ k_bar[b]``.
Rows are grouped by KV head and query tile; a block's tile energy is the sum of
``exp(s - max(s))`` across the tile and all query heads sharing that KV head.
The block is selected when its energy is at least ``alpha`` times the largest
fully visible block energy. Per-row causal visibility is applied before this
reduction, and sink, local-window, and dense-tail blocks are then forced into
the selection.

Query tiles are local to the current prefill chunk, while every visibility
comparison uses ``offset + local_position``. Bottom-right causal chunks thus
select the same blocks as their corresponding tiles in a single-shot prefill.
The forced local interval includes the diagonal block.

The resulting boolean array has shape ``[Hkv, query_tiles, key_blocks]``. This
module is independent of the sparse attention kernel: it defines the selection
math, supports chunking the score tensor by query tiles, and provides a causal
work-density measure for tests and benchmarks.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx


@dataclass(frozen=True)
class FPV2Config:
    """Parameters controlling block selection and mean correction."""

    block_k: int = 128
    tile_q: int = 128
    alpha: float = 0.1
    sink_tokens: int = 256
    window_tokens: int = 384
    # This is request-level only when the input contains the complete request.
    last_n_blocks: int = 2
    mean_correction: bool = True
    score_chunk_tiles: int | None = None
    pack_gqa: bool = False

    def __post_init__(self) -> None:
        if self.block_k <= 0 or self.tile_q <= 0:
            raise ValueError("block_k and tile_q must be positive")
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative")
        if self.sink_tokens < 0 or self.window_tokens < 0:
            raise ValueError("sink_tokens and window_tokens must be non-negative")
        if (
            not isinstance(self.last_n_blocks, int)
            or isinstance(self.last_n_blocks, bool)
            or self.last_n_blocks < 0
        ):
            raise ValueError("last_n_blocks must be a non-negative integer")
        if self.score_chunk_tiles is not None and (
            not isinstance(self.score_chunk_tiles, int)
            or isinstance(self.score_chunk_tiles, bool)
            or self.score_chunk_tiles <= 0
        ):
            raise ValueError("score_chunk_tiles must be a positive integer or None")
        if not isinstance(self.pack_gqa, bool):
            raise ValueError("pack_gqa must be a boolean")


def pool_blocks(x: mx.array, block_size: int = 128) -> mx.array:
    """Mean-pool the penultimate (sequence) axis into fixed-size blocks."""
    if x.ndim < 2:
        raise ValueError("x must have at least sequence and feature dimensions")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    length, head_dim = x.shape[-2:]
    if length % block_size:
        raise ValueError(
            f"sequence length {length} must be divisible by block_size {block_size}"
        )
    shape = (*x.shape[:-2], length // block_size, block_size, head_dim)
    return mx.mean(mx.reshape(x, shape), axis=-2)


def _validate_inputs(
    q: mx.array,
    pooled_k: mx.array,
    config: FPV2Config,
    offset: int,
    query_length: int,
) -> tuple[int, int, int, int]:
    if q.ndim != 3 or pooled_k.ndim != 3:
        raise ValueError("q and pooled_k must have shape [heads, sequence, dim]")
    query_heads, padded_length, head_dim = q.shape
    kv_heads, num_blocks, key_dim = pooled_k.shape
    if not kv_heads or query_heads % kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    if head_dim != key_dim:
        raise ValueError("q and pooled_k head dimensions must match")
    if padded_length % config.tile_q:
        raise ValueError("query length must be padded to a multiple of tile_q")
    if not 0 < query_length <= padded_length:
        raise ValueError("query_length must be in (0, q.shape[-2]]")
    if offset < 0:
        raise ValueError("offset must be non-negative")
    if offset + query_length > num_blocks * config.block_k:
        raise ValueError("queries extend beyond the represented key sequence")
    return query_heads, kv_heads, padded_length, num_blocks


def block_scores_and_select(
    q: mx.array,
    pooled_k: mx.array,
    *,
    scale: float,
    config: FPV2Config = FPV2Config(),
    offset: int = 0,
    query_length: int | None = None,
) -> tuple[mx.array, mx.array | None]:
    """Return selection ``[Hkv, Nt, Nb]`` and optional raw scores.

    ``offset`` is the absolute key position corresponding to query row zero.
    Query input must be tile-padded; ``query_length`` excludes padded rows.
    Raw scores have shape ``[Hkv, G, N, Nb]`` and are returned only when
    ``score_chunk_tiles`` is ``None``. Chunked scoring keeps only each chunk's
    selection mask so its intermediate score memory is bounded.
    """
    valid_length = q.shape[-2] if query_length is None else query_length
    query_heads, kv_heads, padded_length, num_blocks = _validate_inputs(
        q, pooled_k, config, offset, valid_length
    )
    group_size = query_heads // kv_heads
    num_tiles = padded_length // config.tile_q

    grouped_q = mx.reshape(
        q, (kv_heads, group_size, padded_length, q.shape[-1])
    )
    pooled_k32 = pooled_k.astype(mx.float32)
    block_starts = mx.arange(num_blocks, dtype=mx.int32) * config.block_k
    block_ends = block_starts + config.block_k
    sink = block_starts < config.sink_tokens
    chunk_tiles = config.score_chunk_tiles or num_tiles
    selected_chunks: list[mx.array] = []
    raw_scores: mx.array | None = None

    for tile_start in range(0, num_tiles, chunk_tiles):
        tile_end = min(tile_start + chunk_tiles, num_tiles)
        row_start = tile_start * config.tile_q
        row_end = tile_end * config.tile_q
        row_ids = row_start + mx.arange(row_end - row_start, dtype=mx.int32)
        tile_ids = tile_start + mx.arange(tile_end - tile_start, dtype=mx.int32)
        tile_first = offset + tile_ids * config.tile_q
        tile_last = mx.minimum(
            tile_first + config.tile_q, offset + valid_length
        ) - 1
        scores = mx.einsum(
            "hgnd,hbd->hgnb",
            grouped_q[:, :, row_start:row_end, :].astype(mx.float32),
            pooled_k32,
        ) * scale

        query_positions = offset + row_ids
        row_visible = (
            block_starts[None, :] <= query_positions[:, None]
        ) & (row_ids[:, None] < valid_length)
        score_tiles = mx.reshape(
            mx.where(row_visible[None, None, :, :], scores, -mx.inf),
            (
                kv_heads,
                group_size,
                tile_end - tile_start,
                config.tile_q,
                num_blocks,
            ),
        )
        fully_visible = block_ends[None, :] <= tile_last[:, None] + 1
        score_tiles = mx.where(
            fully_visible[None, None, :, None, :], score_tiles, -mx.inf
        )
        tile_max = mx.max(score_tiles, axis=(1, 3, 4), keepdims=True)
        energies = mx.sum(
            mx.where(score_tiles > -mx.inf, mx.exp(score_tiles - tile_max), 0.0),
            axis=(1, 3),
        )
        selected = energies >= config.alpha * mx.max(
            energies, axis=-1, keepdims=True
        )

        visible = block_starts[None, :] <= tile_last[:, None]
        local = block_ends[None, :] > (
            tile_first[:, None] - config.window_tokens
        )
        dense_tail = tile_ids >= max(num_tiles - config.last_n_blocks, 0)
        forced = sink[None, :] | local | dense_tail[:, None]
        selected_chunks.append(
            (selected | forced[None, :, :]) & visible[None, :, :]
        )
        if config.score_chunk_tiles is None:
            raw_scores = scores

    if len(selected_chunks) == 1:
        return selected_chunks[0], raw_scores
    return mx.concatenate(selected_chunks, axis=1), None


def density(
    selected: mx.array,
    *,
    config: FPV2Config = FPV2Config(),
    offset: int = 0,
    query_length: int | None = None,
) -> mx.array:
    """Estimate retained causal block work as a scalar fraction."""
    if selected.ndim != 3:
        raise ValueError("selected must have shape [KV heads, tiles, blocks]")
    kv_heads, num_tiles, num_blocks = selected.shape
    if offset < 0:
        raise ValueError("offset must be non-negative")
    valid_length = num_tiles * config.tile_q if query_length is None else query_length
    if not 0 < valid_length <= num_tiles * config.tile_q:
        raise ValueError("query_length is inconsistent with selected tiles")

    tile_last = mx.minimum(
        offset
        + (mx.arange(num_tiles, dtype=mx.int32) + 1) * config.tile_q,
        offset + valid_length,
    ) - 1
    block_starts = mx.arange(num_blocks, dtype=mx.int32) * config.block_k
    visible = block_starts[None, :] <= tile_last[:, None]
    denominator = mx.sum(visible.astype(mx.float32)) * kv_heads
    return mx.sum(selected.astype(mx.float32)) / denominator
