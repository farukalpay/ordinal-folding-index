# ordinal-folding-index

This repository accompanies the manuscript on ordinal folding dynamics.  The code
is organised as a small library under `library/` and a `main.py` driver script
that runs all available demonstrations.

## Library

```
library/
  __init__.py
  contraction.py   # embedding contraction utility
  benchmarks.py    # fixed-point and LLM benchmarks
```

The `contraction.adjust_embeddings` function implements the contraction
operator described in the paper for post-processing word embeddings.  The
`benchmarks` module exposes two entry points:

- `run_fixed_point_benchmark()` – generates the convergence plot of fixed-point
  solvers.
- `run_llm_benchmark()` – runs the Ordinal Folding Index (OFI) probe on several
  language models.  OpenAI models require a valid API key; HuggingFace models
  require the `transformers` and `torch` packages.

## Usage

Run `python main.py` to execute a short demo of the embedding contraction and
then run both benchmarks.  The fixed-point benchmark produces the figure
`fixed_point_convergence.png` and the LLM benchmark prints a summary table of
OFI scores.  Example output:

```
---------------------------------------------------------
                OFI Benchmark Summary Table
---------------------------------------------------------
 Model            | Factual   | Reasoning | Paradoxical
--------------------------------------------------------
 GPT-3.5 Turbo    | 1.0 ± 0.0 | 1.3 ± 0.5 | 4.0 ± 0.0
 GPT-O3           | 1.0 ± 0.0 | 1.3 ± 0.5 | 3.7 ± 0.9
 GPT-4 (Proxy)    | 1.0 ± 0.0 | 2.3 ± 0.5 | 10.0 ± 0.0
 GPT-2 Large (HF) | 1.0 ± 0.0 | 1.7 ± 0.5 | 4.0 ± 0.8
 DeepSeek (HF)    | 1.0 ± 0.0 | 1.7 ± 0.5 | 3.7 ± 0.5
---------------------------------------------------------
```

The OFI benchmark uses four factual, three reasoning, and three paradoxical
prompts to estimate the index for each model.  Set the `API_KEY` variable in
`library/benchmarks.py` or the `OPENAI_API_KEY` environment variable to run with
the real OpenAI API.  Without a key or without the required libraries, the
benchmark falls back to mock mode.
