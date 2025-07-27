"""Entry point to demonstrate the library utilities."""

import numpy as np
from library.contraction import adjust_embeddings
import library.benchmarks as bm


def demo_contraction() -> None:
    print("\n=== Demonstrating embedding contraction ===")
    E = np.random.randn(5, 20)
    anchors = {0: ([1, 2], [3, 4])}
    adjusted = adjust_embeddings(E, anchors, d1=5, iters=2)
    print("Adjusted vector 0:", adjusted[0])


def main() -> None:
    demo_contraction()
    bm.FORCE_MOCK_MODE = True
    bm.run_fixed_point_benchmark()
    bm.run_llm_benchmark()


if __name__ == "__main__":
    main()

