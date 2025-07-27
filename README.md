# ordinal-folding-index

This repository accompanies the manuscript on ordinal folding dynamics.  All
code has been moved into a small library so it can be reused from scripts or
other projects.

## Library layout

```
library/
    __init__.py
    contraction.py   # embedding contraction operator
    benchmarks.py    # analytic and LLM benchmarks
benchmarks/
    bench1.py        # original standalone benchmark script
main.py              # demonstration entry point
```

The module `library.contraction` implements the contraction operator for
post‑processing word vectors.  `library.benchmarks` provides both the
fixed‑point solver benchmark and the Ordinal Folding Index (OFI) probe for large
language models.

## Running the demos

Execute

```bash
python main.py
```

This will run a tiny contraction example, generate the convergence plot for the
fixed‑point solver benchmark and a mock OFI benchmark using the utilities from
the library.

To run the full benchmarks exactly as in the manuscript you can still invoke the
standalone script

```bash
python benchmarks/bench1.py
```

which requires `numpy`, `matplotlib` and, if you wish to test real language
models, `openai`, `transformers` and `torch`.

