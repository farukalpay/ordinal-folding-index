import numpy as np

"""Utilities for adjusting word embeddings using a contraction operator.

This module implements a simplified version of the fixed-point method
outlined in the manuscript. It treats the embedding space as a product of
a hyperbolic component (implemented with the Poincaré ball model) and a
Euclidean tail. The main entry point is ``adjust_embeddings`` which updates
selected vectors based on positive and negative anchor sets.
"""

# ---------------------------------------------------------------------------
# Basic operations on the Poincaré ball
# ---------------------------------------------------------------------------

def _mobius_add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    xy = np.sum(x * y, axis=-1, keepdims=True)
    x_sq = np.sum(x * x, axis=-1, keepdims=True)
    y_sq = np.sum(y * y, axis=-1, keepdims=True)
    denom = 1 + 2 * xy + x_sq * y_sq
    return ((1 + 2 * xy + y_sq) * x + (1 - x_sq) * y) / np.clip(denom, 1e-8, None)

def _mobius_scalar_mul(r: float, x: np.ndarray) -> np.ndarray:
    x_norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return np.tanh(r * np.arctanh(np.clip(x_norm, 0.0, 0.999))) * x / np.clip(x_norm, 1e-8, None)

def _exp_map(x: np.ndarray, v: np.ndarray) -> np.ndarray:
    v_norm = np.linalg.norm(v, axis=-1, keepdims=True)
    second_term = _mobius_scalar_mul(np.tanh(v_norm / 2) / np.clip(v_norm, 1e-8, None), v)
    return _mobius_add(x, second_term)

def _log_map(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    diff = _mobius_add(-x, y)
    diff_norm = np.linalg.norm(diff, axis=-1, keepdims=True)
    return (2 / np.clip(1 - np.sum(x * x, axis=-1, keepdims=True), 1e-8, None)) * np.arctanh(np.clip(diff_norm, 0.0, 0.999)) * diff / np.clip(diff_norm, 1e-8, None)

# ---------------------------------------------------------------------------
# Contraction step
# ---------------------------------------------------------------------------

def _hyperbolic_contraction(x: np.ndarray, m_pos: np.ndarray, m_neg: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    v_pos = _log_map(x, m_pos)
    v_neg = _log_map(x, m_neg)
    inner = float(np.dot(v_neg, v_pos))
    update = alpha * v_pos - beta * inner * v_neg / np.clip(np.linalg.norm(v_neg), 1e-8, None)
    return _exp_map(x, update)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def adjust_embeddings(E: np.ndarray, anchor_sets: dict, d1: int = 10, alpha: float = 0.4, beta: float = 0.1, iters: int = 20) -> np.ndarray:
    """Applies the contraction operator to selected embeddings.

    Parameters
    ----------
    E : ndarray of shape (V, d)
        Original embeddings.
    anchor_sets : dict
        Maps an index ``i`` to a tuple ``(A_plus, A_minus)`` of lists containing
        positive and negative anchor indices for ``i``.
    d1 : int, optional
        Dimension of the hyperbolic component.
    alpha : float, optional
        Attraction strength toward the positive centroid. Must satisfy ``0 < alpha < 0.5``.
    beta : float, optional
        Repulsion strength from the negative centroid. Use ``0 < beta < alpha``.
    iters : int, optional
        Number of contraction iterations to perform.

    Returns
    -------
    ndarray
        The adjusted embeddings of the same shape as ``E``.
    """
    d = E.shape[1]
    assert d1 <= d, "d1 must not exceed embedding dimension"
    E_new = E.copy()

    for i, (A_pos, A_neg) in anchor_sets.items():
        if not A_pos or not A_neg:
            continue
        x = E_new[i]
        x_h, x_e = x[:d1], x[d1:]
        m_pos_h = E_new[A_pos, :d1].mean(axis=0)
        m_neg_h = E_new[A_neg, :d1].mean(axis=0)
        m_pos_e = E_new[A_pos, d1:].mean(axis=0)
        m_neg_e = E_new[A_neg, d1:].mean(axis=0)

        for _ in range(iters):
            x_h = _hyperbolic_contraction(x_h, m_pos_h, m_neg_h, alpha, beta)
            v_pos_e = m_pos_e - x_e
            v_neg_e = m_neg_e - x_e
            inner_e = float(np.dot(v_neg_e, v_pos_e))
            x_e = x_e + alpha * v_pos_e - beta * inner_e * v_neg_e / np.clip(np.linalg.norm(v_neg_e), 1e-8, None)

        E_new[i] = np.concatenate([x_h, x_e])

    return E_new

__all__ = ["adjust_embeddings"]
