#!/usr/bin/env python3
# simulate_news_data.py
# ---------------------------------------------------------------------
# Produce synthetic NEWS data in the same format as the MIMIC pipeline.
# ---------------------------------------------------------------------
#!/usr/bin/env python3
# simulate_news_data.py
# ---------------------------------------------------------------------
# Synthetic NY-Times “NEWS” data in the same on-disk format as the
# MIMIC pipeline, but with the NEWS-specific treatment and outcome
# mechanisms described in the original code base.
# ---------------------------------------------------------------------
#https://github.com/rtealwitter/naturalexperiments
import os
import gzip
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from src.config import *
# After loading X_all (bag-of-words)
from sklearn.decomposition import LatentDirichletAllocation

from sklearn.model_selection import train_test_split
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import normalize

import argparse


# ---------------------------------------------------------------------
# 1 · Data loading
# ---------------------------------------------------------------------
NUMBER_OF_FEATURES = 3477           # size of the fixed vocabulary
N_DOCS              = 5000          # first 5 000 articles


import argparse


parser = argparse.ArgumentParser(
    description="Train CCS Counterfactual Net with fixed defaults.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
        "--seed", default=527, help="Dataset",
    )

parser.parse_args()

def load_docword_matrix(path: str,
                        num_docs: int = N_DOCS,
                        num_words: int = NUMBER_OF_FEATURES) -> np.ndarray:
    """
    Convert the compressed 'docword.nytimes.txt.gz' bag-of-words file
    into a dense (num_docs × num_words) float32 matrix.
    """
    with gzip.open(path, "rt") as f:
        _ = [f.readline() for _ in range(3)]               # discard header
        matrix = np.zeros((num_docs, num_words), np.float32)
        for ln in f:
            doc_id, word_id, cnt = map(int, ln.split())
            if doc_id <= num_docs and word_id <= num_words:
                matrix[doc_id - 1, word_id - 1] = cnt
    return matrix

# ---------------------------------------------------------------------
# 2 · Latent projection vectors  (v₁, v₂, v₃)
# ---------------------------------------------------------------------
def init_v(random_seed: int, dim_cov: int):
    """
    Draw three Gaussian vectors and L2-normalise them.
    Returns v1, v2, v3   each shaped (dim_cov,).
    """
    rng          = np.random.default_rng(SEED)
    v1p, v2p, v3p = rng.normal(0.0, 1.0, size=(3, dim_cov))
    v1, v2, v3    = [v / np.linalg.norm(v, 2) for v in (v1p, v2p, v3p)]
    return v1.astype(np.float32), v2.astype(np.float32), v3.astype(np.float32)

# ---------------------------------------------------------------------
# 3 · NEWS-style treatment generator
# ---------------------------------------------------------------------
def compute_beta(alpha: np.ndarray, optimal_dosage: np.ndarray) -> np.ndarray:
    """
    Closed-form β so that the Beta(α, β) mode equals optimal_dosage.
    Edge cases (≈0 or ≥1) fall back to β = 1.
    """
    beta = np.where(
        (optimal_dosage <= 0.001) | (optimal_dosage >= 1.0),
        1.0,
        (alpha - 1.0) / optimal_dosage + (2.0 - alpha)
    )
    return beta

def simulate_t(x: np.ndarray,
               v2: np.ndarray,
               v3: np.ndarray,
               selection_bias: float = 2.0) -> np.ndarray:
    """
    NEWS assignment:  t ∼ Beta(α, β) where the mode depends on
    ratio = (v₃·x) / (2 v₂·x).  Returns shape (n_samples, 1).
    """
    optimal_dosage = (x @ v3) / (2.0 * (x @ v2) + 1e-8)
    alpha          = np.full_like(optimal_dosage, selection_bias, np.float32)
    beta           = compute_beta(alpha, optimal_dosage)

    t = np.random.beta(alpha, beta).astype(np.float32)
    t = np.where(optimal_dosage <= 0.001, 1.0 - t, t)          # flip edge case
    return t.reshape(-1, 1)

# ---------------------------------------------------------------------
# 4 · NEWS-style outcome surface
# ---------------------------------------------------------------------
def simulate_y(t: np.ndarray,
               x: np.ndarray,
               v1: np.ndarray,
               v2: np.ndarray,
               v3: np.ndarray,
               noise_sd: float = 0.2,
               eps: float = 1e-8) -> np.ndarray:
    """
    y = 10 · (v₁·x + sin(π · ratio · t)) + ε,    ε ~ 𝒩(0, noise_sd²)
    ratio = (v₂·x)/(v₃·x); ratio→0 if denominator tiny.
    """
    denom = x @ v3
    ratio = np.where(np.abs(denom) < eps, 0.0, (x @ v2) / denom)
    core  = (x @ v1) + np.sin(np.pi * ratio * t.squeeze())
    y     = 10.0 * core #+ np.random.normal(0.0, noise_sd, size=len(core))
    return y.astype(np.float32)

# ---------------------------------------------------------------------
# 5 · Convenience wrapper for a full sample (t, x, y)
# ---------------------------------------------------------------------
def simulate_data(x: np.ndarray,
                  v1: np.ndarray,
                  v2: np.ndarray,
                  v3: np.ndarray,
                  selection_bias: float = 2.0,
                  noise_sd: float = 0.2) -> np.ndarray:
    """
    Stack [t | x | y] into one array shaped (n_samples, n_features + 2)
    so the output format mirrors MIMIC.
    """
    t = simulate_t(x, v2, v3, selection_bias)
    y = simulate_y(t, x, v1, v2, v3, noise_sd)


    return np.hstack((x, t, y.reshape(-1, 1))).astype(np.float32)

# ---------------------------------------------------------------------
# 6 · Driver
# ---------------------------------------------------------------------
def main():

    # ---------- configuration ----------
    data_dir        = "data/news"
    docword_gz      = os.path.join(data_dir, "docword.nytimes.txt.gz")
    random_seed     = SEED #42#592#468#42#468 #592, 468, 345
    selection_bias  = 2#2.0          # α in the Beta distribution
    noise_sd        = 0.2
    n_grid_1dim     = 11
    # -----------------------------------

    # 6-a · load covariate matrix
    X_all = load_docword_matrix(docword_gz)                 # (5000, 3477)

    # lda = LatentDirichletAllocation(n_components=10, random_state=0)
    # X_all = lda.fit_transform(X_all)

    # X_all = normalize(X_all, norm='l1', axis=1)

    # Load data
    X_all = load_docword_matrix(docword_gz)

    # Split into train and test BEFORE fitting LDA
    X_train_raw, X_test_raw = train_test_split(X_all, test_size=0.2, random_state=42)

    # Fit LDA on training data only
    lda = LatentDirichletAllocation(n_components=10, random_state=0)
    X_train = lda.fit_transform(X_train_raw)
    X_test = lda.transform(X_test_raw)  # Use transform, not fit_transform

    # Normalize both splits
    X_train = normalize(X_train, norm='l1', axis=1)
    X_test = normalize(X_test, norm='l1', axis=1)

    # Then continue using X_all_topics instead of X_all
    #X_train, X_test = X_all_topics[idx_train], X_all_topics[idx_test]

    # per-document L1 normalisation (bag-of-words → relative term freq.)


    # 6-b · latent vectors
    v1, v2, v3 = init_v(random_seed, X_train.shape[1])

    # # 6-c · split into train / test (80 / 20)
    # idx_train, idx_test = train_test_split(
    #     np.arange(X_all.shape[0]),
    #     test_size=0.2,
    #     random_state=random_seed,
    #     shuffle=True,
    # )
    # X_train, X_test = X_all[idx_train], X_all[idx_test]

    # 6-d · simulate t and y
    data_train = simulate_data(
        X_train, v1, v2, v3,
        selection_bias=selection_bias,
        noise_sd=noise_sd,
    )
    data_test = simulate_data(
        X_test, v1, v2, v3,
        selection_bias=selection_bias,
        noise_sd=noise_sd,
    )

    # 6-e · evaluation grid (potential-outcome curves, noise-free)
    t_grid = np.linspace(0.01, 1.0, n_grid_1dim)[:, None]   # (G,1)


    y_eval_test = simulate_y(
        np.tile(t_grid, (X_test.shape[0], 1)),
        np.repeat(X_test, t_grid.shape[0], axis=0),
        v1, v2, v3,
        noise_sd=0.0,
    )
    data_eval_test = np.hstack(
        (np.tile(t_grid, (X_test.shape[0], 1)),
         y_eval_test.reshape(-1, 1))
    ).reshape(-1, n_grid_1dim, 2)      # each record: G × (t,y)

    # 6-f · persist artefacts in the same layout as MIMIC
    os.makedirs(data_dir, exist_ok=True)
    np.save(os.path.join(data_dir, "train.npy"),     data_train)
    np.save(os.path.join(data_dir, "test.npy"),      data_test)
    np.save(os.path.join(data_dir, "eval_test.npy"), data_eval_test)
    np.save(os.path.join(data_dir, "v_vector.npy"),
            np.stack([v1, v2, v3], axis=0))

    pd.to_pickle({
        "random_seed":     random_seed,
        "n_grid_1dim":     n_grid_1dim,
        "dim_treat":       1,
        "selection_bias":  selection_bias,
        "noise_sd":        noise_sd,
    }, os.path.join(data_dir, "info.pkl"))

    print("✔  Saved train.npy, test.npy, eval_test.npy, v_vector.npy, info.pkl")

# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
