import numpy as np
from library import adjust_embeddings, run_fixed_point_benchmark, run_llm_benchmark


def demo_contraction():
    E = np.random.randn(5, 20)
    anchors = {0: ([1, 2], [3, 4])}
    E_new = adjust_embeddings(E, anchors, d1=5, iters=2)
    print("Adjusted embeddings shape:", E_new.shape)


def main():
    demo_contraction()
    run_fixed_point_benchmark()
    run_llm_benchmark()


if __name__ == "__main__":
    main()
