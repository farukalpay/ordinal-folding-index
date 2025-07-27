# ordinal-folding-index

This repository accompanies the manuscript on ordinal folding dynamics. It now
provides a small Python library and a demonstration script.

## Library structure

```
library/
  embedding_contraction.py  # post-processing for word vectors
  bench1.py                 # fixed-point and OFI benchmarks
main.py                     # runs a short demo
```

`embedding_contraction.py` implements the contraction operator described in the
manuscript for adjusting embeddings. `bench1.py` exposes the benchmark
functions originally contained in `benchmarks/bench1.py`.

## Quick start

1. Install the Python dependencies. Only `numpy` and `matplotlib` are required
   for the demo. Optionally install `openai`, `transformers`, and `torch` to run
   the full OFI benchmark.

```bash
pip install numpy matplotlib  # plus openai/transformers/torch if desired
```

2. Execute the demo which exercises the library and runs both benchmarks
   (using mock data if API libraries are unavailable):

```bash
python main.py
```

The script prints progress messages, produces the convergence plot
`fixed_point_convergence.png`, and displays a summary table of OFI scores.

The original benchmark script remains under `benchmarks/bench1.py` for
reference; the library exposes the same functionality for reuse in other
projects.
