# Models

Store each downloaded MLX model in its own directory. Model contents are
intentionally ignored by Git; only this guide is tracked.

Download an MLX-format model with the project CLI:

```sh
uv run fpv2-download mlx-community/Qwen3-4B-Thinking-2507-8bit
```

By default this creates `models/Qwen3-4B-Thinking-2507-8bit`. Use `--name` to
choose another directory name, `--revision` to pin a model revision, and
`--token` for a gated repository. Existing local directories and Hugging Face
repository IDs can also be passed directly to `fpv2-generate` without using
this directory. The generator defaults to this Hugging Face repository when no
model argument is provided.

```sh
uv run fpv2-generate -p "Summarize sparse attention."
```

FlashPrefillV2-MLX hooks MLX scaled dot-product attention and does not require
model-specific source code. The current sparse path supports batch-one causal
prefill, grouped-query attention, float16 or bfloat16 Q/K/V, and head dimensions
64, 96, 128, or 256. Other attention calls fall back to MLX's stock operator.

Model weights remain subject to their upstream licenses and usage terms. Check
the model card before downloading or redistributing a snapshot, and account for
both weight storage and MLX's runtime memory when selecting a quantization.

Only full causal attention layers use FPV2. Sliding-window attention layers,
SSM and linear-attention layers, decode steps, and prompts shorter than
`--min-tokens` continue through the model's normal dense path.
