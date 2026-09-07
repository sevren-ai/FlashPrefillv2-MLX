# AGENTS.md

FlashPrefillv2-MLX is a Python and MLX implementation of FlashPrefill V2 for
Apple silicon. Follow the current code and project configuration rather than
assuming a fixed layout or toolchain.

## References

- Paper: [FlashPrefill V2: Block-Sparse Prefill Attention for Long-Context LLM
Serving](https://arxiv.org/abs/2608.19758)
- The original FlashPrefill V2 repository can clarify algorithms, terminology,
and expected behavior. Do not mechanically port CUDA or PyTorch code; adapt
the design to MLX and Metal.

## Development

- Read the relevant code, tests, `README.md`, and `pyproject.toml` before making
changes. These are the source of truth for supported commands and conventions.
- Prefer small, focused changes. Avoid speculative abstractions and compatibility
layers for unreleased behavior.
- Use type annotations for public APIs and document non-obvious tensor shapes,
layouts, dtypes, and numerical assumptions.
- Validate shapes and dtypes at public boundaries with useful error messages.
- Add dependencies only when existing dependencies are insufficient.
- Update tests and documentation with behavior changes. Run the narrowest
relevant checks first, then the documented full suite when practical.



## MLX

- Keep compute paths in MLX. Avoid unnecessary NumPy, Python-scalar, or other
framework conversions.
- Respect lazy execution. Use `mx.eval(...)` at intentional synchronization
points, especially for correctness checks and benchmarks, not in hot paths.
- Be explicit about precision and accumulation dtypes. Do not silently cast
public inputs.
- Avoid materializing full attention matrices or expanded tensors when a tiled,
fused, or streaming implementation is intended.
- Start with a clear MLX reference implementation. Add custom Metal kernels or
compilation only after correctness is established and benchmarks identify a
meaningful bottleneck.
- Custom kernels must state supported shapes and dtypes, handle boundaries, and
remain testable against the reference implementation.



## Correctness

- Compare optimized paths with a straightforward reference using
dtype-appropriate tolerances.
- Test causal masking, block boundaries, non-divisible sequence lengths,
grouped-query head mapping, supported dtypes, and invalid inputs as relevant.
- For sparse attention, test selected-block validity and dense equivalence when
every block is selected, not only final output values.
- Use deterministic seeds and small routine test cases. Keep large or
hardware-specific performance tests separate.



## Performance

- Preserve a readable correctness baseline before optimizing.
- Synchronize MLX execution around timed regions and separate compilation and
warm-up from steady-state measurements.
- Benchmark representative and boundary shapes. Record hardware, MLX version,
dtype, warm-up, iterations, and synchronization method.
- Support performance claims with reproducible measurements; report memory and
numerical tradeoffs where relevant.
