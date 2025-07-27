import numpy as np
from library.contraction import adjust_embeddings
from library.benchmarks import run_fixed_point_benchmark, run_llm_benchmark


def demo_adjust_embeddings():
    print("\n--- Demo: adjust_embeddings ---")
    E = np.random.randn(5, 20)
    anchor_sets = {0: ([1, 2], [3, 4])}
    E_adj = adjust_embeddings(E, anchor_sets, d1=5, iters=5)
    print("Adjusted vector for index 0:", E_adj[0])


def main():
    demo_adjust_embeddings()
    run_fixed_point_benchmark()
    run_llm_benchmark()


if __name__ == "__main__":
    main()
