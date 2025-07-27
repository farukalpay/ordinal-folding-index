import numpy as np
from library.embedding_contraction import adjust_embeddings
from library.bench1 import run_fixed_point_benchmark, run_llm_benchmark


def main():
    print("Testing adjust_embeddings with toy data...")
    E = np.random.randn(5, 20)
    anchor_sets = {0: ([1, 2], [3, 4])}
    E_new = adjust_embeddings(E, anchor_sets, d1=5, iters=1)
    print("Updated embedding shape:", E_new.shape)

    print("\nRunning fixed-point benchmark (short demonstration)...")
    run_fixed_point_benchmark()

    print("\nRunning LLM benchmark (mock mode if APIs unavailable)...")
    run_llm_benchmark()


if __name__ == "__main__":
    main()
