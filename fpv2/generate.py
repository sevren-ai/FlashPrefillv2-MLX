"""Stream text from a local or Hugging Face MLX model with FlashPrefill V2."""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

from fpv2 import FPV2Config, block_sparsity, disable, enable


DEFAULT_MODEL = "mlx-community/Qwen3-4B-Thinking-2507-8bit"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model",
        nargs="?",
        default=DEFAULT_MODEL,
        help=f"Local model path or Hugging Face repo (default: {DEFAULT_MODEL})",
    )
    prompt = parser.add_mutually_exclusive_group()
    prompt.add_argument("-p", "--prompt", help="Prompt text (default: Hello)")
    prompt.add_argument(
        "--prompt-file", metavar="PATH", help="Read the prompt from PATH, or - for stdin"
    )
    parser.add_argument(
        "--chat-template",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Wrap the prompt as a user message (default: enabled)",
    )
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--min-tokens", type=int, default=4_096)
    parser.add_argument("--no-corr", action="store_true", help="Disable mean correction")
    parser.add_argument("--dense", action="store_true", help="Use stock dense attention")
    parser.add_argument("--prefill-step-size", type=int, default=2_048)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.max_tokens <= 0 or args.prefill_step_size <= 0:
        raise ValueError("--max-tokens and --prefill-step-size must be positive")
    if args.min_tokens <= 0:
        raise ValueError("--min-tokens must be positive")

    prompt_text = args.prompt if args.prompt is not None else "Hello"
    if args.prompt_file is not None:
        prompt_text = (
            sys.stdin.read()
            if args.prompt_file == "-"
            else Path(args.prompt_file).read_text(encoding="utf-8")
        )

    print(f"Loading model {args.model}...", file=sys.stderr, flush=True)
    load_start = time.perf_counter()
    model, tokenizer = load(args.model)
    print(f"Model loaded in {time.perf_counter() - load_start:.3f} s", file=sys.stderr)
    prompt: str | list[int] = prompt_text
    if args.chat_template:
        if not getattr(tokenizer, "has_chat_template", False):
            raise ValueError("the tokenizer has no chat template")
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=True,
            add_generation_prompt=True,
        )

    prompt_tokens = (
        len(prompt)
        if isinstance(prompt, list)
        else len(tokenizer.encode(prompt, add_special_tokens=False))
    )
    print(f"Prompt: {prompt_tokens} tokens", file=sys.stderr)
    if args.dense:
        print("Attention: dense MLX SDPA", file=sys.stderr)
    else:
        print(
            f"Attention: FPV2 alpha={args.alpha:g}, "
            f"correction={not args.no_corr}, min_tokens={args.min_tokens}",
            file=sys.stderr,
        )

    disable()
    last = None
    try:
        if not args.dense:
            enable(
                FPV2Config(alpha=args.alpha, mean_correction=not args.no_corr),
                min_tokens=args.min_tokens,
                collect_stats=True,
                total_tokens=prompt_tokens,
            )
        sampler = make_sampler(temp=args.temperature)
        for response in stream_generate(
            model,
            tokenizer,
            prompt,
            max_tokens=args.max_tokens,
            sampler=sampler,
            prefill_step_size=args.prefill_step_size,
        ):
            print(response.text, end="", flush=True)
            last = response
    finally:
        disable()
    print()
    if last is not None:
        seconds = last.prompt_tokens / last.prompt_tps
        sparsity = "0%" if args.dense else f"{block_sparsity():.1%}"
        print(
            f"prefill: {last.prompt_tokens} tokens, {seconds:.3f} s, "
            f"{last.prompt_tps:.1f} tok/s, {last.peak_memory:.3f} GB peak, "
            f"block sparsity: {sparsity}",
            file=sys.stderr,
        )
        print(
            f"decode: {last.generation_tokens} tokens, "
            f"{last.generation_tps:.1f} tok/s",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
