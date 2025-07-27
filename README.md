# ordinal-folding-index

This repository accompanies the manuscript on ordinal folding dynamics. The
codebase is now organized as a small Python library under the `library/`
directory.

## Library overview

- `library.contraction.adjust_embeddings` implements the contraction operator
  for post-processing word vectors.
- `library.benchmarks` exposes `run_fixed_point_benchmark()` and
  `run_llm_benchmark()` which wrap the analytic and LLM benchmarks originally
  provided in `benchmarks/bench1.py`.

## Running the demo

A convenience script `main.py` demonstrates both the contraction operator and
the benchmarks. Execute it from the repository root:

```bash
python main.py
```

By default the LLM benchmark runs in mock mode unless the `openai` package is
installed and a valid `OPENAI_API_KEY` environment variable is available.
Support for local HuggingFace models requires `transformers` and `torch`.

The fixed-point benchmark saves `fixed_point_convergence.png` and prints an OFI
summary table.

## Requirements

The core utilities depend only on `numpy`. The benchmarks additionally require
`matplotlib` and optionally `openai`, `transformers`, and `torch`.

```bash
pip install numpy matplotlib openai transformers torch
```

Install only the packages you need.

## Legacy scripts

The original benchmark script remains in `benchmarks/bench1.py` and can still be
run directly if desired:

```bash
python benchmarks/bench1.py
```
