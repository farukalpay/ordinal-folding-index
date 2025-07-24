# ordinal-folding-index

This repository accompanies the manuscript on ordinal folding dynamics. The
`benchmarks/bench1.py` script runs two demonstrations:

1. A fixed-point solver benchmark producing a convergence plot.
2. An Ordinal Folding Index (OFI) probe for large language models.

The OFI benchmark supports OpenAI models and now includes optional support for
local HuggingFace models such as **GPT-2 Large**. To use GPT-2 locally, ensure
that `transformers` and `torch` are installed.

Running `python benchmarks/bench1.py` generates `fixed_point_convergence.png`
and prints a summary table of OFI scores:

```
---------------------------------------------------------
                OFI Benchmark Summary Table
---------------------------------------------------------
 Model            | Factual    | Reasoning | Paradoxical 
---------------------------------------------------------
 GPT-2 Large (HF) | 1.0 ± 0.0  | 1.3 ± 0.5 | 4.0 ± 0.0   
 GPT-3.5 Turbo    | 10.0 ± 0.0 | 7.3 ± 3.1 | 9.3 ± 0.9   
 GPT-4 (Proxy)    | 10.0 ± 0.0 | 9.7 ± 0.5 | 10.0 ± 0.0  
---------------------------------------------------------
```

The benchmark uses four factual, three reasoning, and three paradoxical prompts
to estimate the OFI for each model. 
