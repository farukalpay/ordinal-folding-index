# ordinal-folding-index

This repository accompanies the manuscript on ordinal folding dynamics. The
`benchmarks/bench1.py` script runs two demonstrations:

1. A fixed-point solver benchmark producing a convergence plot.
2. An Ordinal Folding Index (OFI) probe for large language models.

The OFI benchmark supports OpenAI models and now includes optional support for
local HuggingFace models such as **GPT-2 Large** and **DeepSeek**. It also
benchmarks OpenAI's latest **GPT-O3** model. To use the HuggingFace models
locally, ensure that `transformers` and `torch` are installed.

Running `python benchmarks/bench1.py` generates `fixed_point_convergence.png`
and prints a summary table of OFI scores:

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

The benchmark uses four factual, three reasoning, and three paradoxical prompts
to estimate the OFI for each model.

The repository also provides `embedding_contraction.py`, which implements the
contraction operator discussed in the manuscript for post-processing word
embeddings.  See the module's docstring for usage details.
