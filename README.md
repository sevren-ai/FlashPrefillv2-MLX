# FlashPrefillv2-MLX

FlashPrefillv2-MLX accelerates long-context causal prefill on Apple silicon by
replacing dense attention with block-sparse attention while preserving a pooled
correction for omitted blocks. The baseline is official MLX-LM prefill for one
request at the same prompt length, with the model loaded once per process.

This project implements [FlashPrefill V2](https://arxiv.org/abs/2608.19758) for
MLX. The paper authors' Hopper implementation is available at
[qhfan/FlashPrefillv2](https://github.com/qhfan/FlashPrefillv2).

## Results

Measured on three Apple silicon systems with MLX 0.32.2 and MLX-LM 0.31.3. Times are means across attempts for each dense/FPV2 configuration.


| System | GPU cores | Unified memory | macOS  | Attempts |
| ------ | --------- | -------------- | ------ | -------- |
| M1 Pro | 14        | 16 GB          | 26.4.1 | 8        |
| M1 Max | 32        | 64 GB          | 26.7   | 3        |
| M4 Pro | 16        | 24 GB          | 26.6   | 3        |


### End To End

Qwen3-4B-Thinking-2507-8bit, `alpha=0.1`, 2,048-token prefill chunks, and the
checked-in NIAH prompts:


| System       | Prompt | Tokens | Stock     | FPV2      | Speedup | FPV2 peak |
| ------------ | ------ | ------ | --------- | --------- | ------- | --------- |
| M1 Pro 16 GB | 8k     | 8,192  | 31.081 s  | 27.729 s  | 1.121x  | 6.322 GB  |
| M1 Pro 16 GB | 16k    | 16,383 | 76.850 s  | 57.712 s  | 1.332x  | 7.784 GB  |
| M1 Pro 16 GB | 32k    | 32,768 | 214.603 s | 123.734 s | 1.734x  | 10.684 GB |
| M1 Pro 16 GB | 64k    | 65,535 | not run   | not run   | -       | -         |
| M1 Max 64 GB | 8k     | 8,192  | 15.451 s  | 13.625 s  | 1.134x  | 6.322 GB  |
| M1 Max 64 GB | 16k    | 16,383 | 40.135 s  | 30.759 s  | 1.305x  | 7.796 GB  |
| M1 Max 64 GB | 32k    | 32,768 | 108.046 s | 65.147 s  | 1.659x  | 10.697 GB |
| M1 Max 64 GB | 64k    | 65,535 | 334.917 s | 147.660 s | 2.268x  | 17.463 GB |
| M4 Pro 24 GB | 8k     | 8,192  | 15.014 s  | 13.393 s  | 1.121x  | 6.322 GB  |
| M4 Pro 24 GB | 16k    | 16,383 | 37.049 s  | 27.339 s  | 1.355x  | 7.793 GB  |
| M4 Pro 24 GB | 32k    | 32,768 | 104.604 s | 58.644 s  | 1.784x  | 10.697 GB |
| M4 Pro 24 GB | 64k    | 65,535 | 346.900 s | 134.330 s | 2.582x  | 17.463 GB |


Block sparsity was identical across systems: 55.8% at 8k, 73.0% at 16k, 83.5% at 32k, and 88.6% at 64k. Dense and FPV2 runs both retrieved the correct key in every attempt.

End-to-end gains are bounded because attention is roughly 40% of Qwen3-4B-Thinking-2507 prefill FLOPs at 16k; the MLP dominates, so even free attention would cap the gain near 1.7x. Much larger gains reported by the paper at 128k apply where attention dominates and retained density is much lower.

### Accuracy

The static prompts use different Project Gutenberg books and fixed,
non-memorable keys placed halfway through each excerpt. Each prompt was tested
with real greedy generation using the model's chat template, checking whether
both stock attention and FPV2 retrieved the corresponding key.


| Prompt                 | Project Gutenberg text             | Key      |
| ---------------------- | ---------------------------------- | -------- |
| `niah_example_8k.txt`  | *Alice's Adventures in Wonderland* | `739184` |
| `niah_example_16k.txt` | *Pride and Prejudice*              | `582617` |
| `niah_example_32k.txt` | *Moby-Dick*                        | `406953` |
| `niah_example_64k.txt` | *War and Peace*                    | `871245` |




## Method

The index stage mean-pools keys into 128-token blocks, scores each query against
the pooled keys, and sums exponentiated scores over every 128 packed query rows
and the query heads sharing a KV head. Thus a GQA group of size `G` uses
`128 / G` query positions per selection tile. Only key blocks fully visible at
the tile's final row contribute to energies; the partially visible diagonal is
forced separately. A block is retained when its energy is at least `alpha`
times the strongest block. Sink blocks and three preceding blocks plus the
diagonal are always retained. The final two query tiles are dense by default.
For chunked prefill, the hook applies this rule only when the key length reaches
the `total_tokens` request boundary; without a known boundary it conservatively
disables the dense tail. Long inputs can score fixed groups of query tiles to
bound transient memory without changing selection.

The Metal stage follows the FlashAttention-2 online-softmax structure using
Steel `simdgroup_matrix` fragments. It visits only retained blocks, keeps Q
resident in threadgroup memory, aliases K and V storage, and accumulates in
float32 with base-2 exponentials. Safely pruned blocks enter the same softmax as
pooled key/value entries with a `+log2(128)` logit shift, exactly representing
the block-size weight in the mean correction.

## Implementation Notes

- Apple GPUs have no FP8 matrix multiplication, so inputs are float16 or
bfloat16 with float32 accumulation.
- Hopper-specific TMA, warp specialization, and ping-pong scheduling are not
used.
- Block lists use fixed-width padding instead of data-dependent CSR storage,
preserving lazy MLX execution without host synchronization.
- Correction eligibility is shared by each 128-packed-row query tile.
- PackGQA is implemented but disabled by default. At alpha=0.1 it measured  
156.130 ms versus 153.491 ms flat, 1.7% slower because this kernel is  
compute-bound and packing overhead exceeds the saved KV traffic.

## Install

```sh
uv sync --extra test
```



## CLI

```sh
uv run fpv2-download mlx-community/Qwen3-4B-Thinking-2507-8bit
uv run fpv2-generate -p "Summarize sparse attention." --alpha 0.1
uv run fpv2-generate --prompt-file prompts/niah_example_8k.txt
```

Use `--dense` for stock attention, `--no-corr` to disable mean correction, and
`--prefill-step-size` to control MLX-LM prompt chunking. The default model is
`mlx-community/Qwen3-4B-Thinking-2507-8bit`, and chat templating is enabled by
default; use `--no-chat-template` for raw completion. Generated text goes to
stdout; model-load, prefill, memory, block sparsity, and decode metrics go to
stderr. Block sparsity is the fraction of visible causal blocks skipped across
supported prefill calls; `--dense` reports 0%.

## Layout

```text
fpv2/               selection, reference, Metal kernel, hook, and CLIs
fpv2/metal/steel/   MLX 0.32.2 Steel headers and provenance
tests/              exact, reference, kernel, offset, and hook coverage
prompts/            ready-to-run long-context examples
models/             ignored local model snapshots
```



## Tests

```sh
uv run pytest -q
```
