# ordinal-folding-index

This repository accompanies the manuscript on ordinal folding dynamics. The
`benchmarks/bench1.py` script runs two demonstrations:

1. A fixed-point solver benchmark producing a convergence plot.
2. An Ordinal Folding Index (OFI) probe for large language models.

The OFI benchmark supports OpenAI models and now includes optional support for
local HuggingFace models such as **GPT-2 Large**. To use GPT-2 locally, ensure
that `transformers` and `torch` are installed.
