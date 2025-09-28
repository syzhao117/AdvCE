import numpy as np
from numpy.random import default_rng
from scipy.stats import beta as beta_dist


def generate_tcga_outcomes(x_df, t_vec, seed=42):
    """
    Generate outcomes for a batch of samples using the TCGA outcome function.

    Parameters:
    - x_df: pandas DataFrame of shape (n_samples, n_features)
    - t_vec: numpy array or list of treatment values, shape (n_samples,)
    - seed: optional int, for reproducibility

    Returns:
    - y: numpy array of shape (n_samples,)
    - v1, v2, v3: weight vectors used
    """
    if seed is not None:
        np.random.seed(seed)

    X = x_df.to_numpy()
    t_vec = np.asarray(t_vec).flatten()
    
    if X.shape[0] != t_vec.shape[0]:
        raise ValueError("Mismatch: x has %d rows but t has %d values" % (X.shape[0], t_vec.shape[0]))

    d = X.shape[1]

    def sample_and_normalize():
        v = np.random.randn(d)
        return v / np.linalg.norm(v)

    v1 = sample_and_normalize()
    v2 = sample_and_normalize()
    v3 = sample_and_normalize()

    # Dot products per sample
    dot1 = X @ v1         # shape (n_samples,)
    dot2 = X @ v2         # shape (n_samples,)
    dot3 = X @ v3         # shape (n_samples,)

    y = 10 * (dot1 + 12 * dot2 * t_vec - 12 * dot3 * t_vec**2)
    return y, v1, v2, v3

def assign_treatments_tcga(df, alpha=2.0, beta_const=2.0, seed=42):
    rng = default_rng(seed)
    n = len(df)
    t = rng.beta(alpha, beta_const, size=n).astype(np.float32)
    return t, None, None




def generate_news_outcomes(x_df, t_vec, seed=42):
    """
    Generate outcomes for a batch of samples using the NEWS outcome function:
        y = 10 * (v1^T x + sin((v2^T x / v3^T x) * π * t))

    Parameters:
    - x_df: pandas DataFrame of shape (n_samples, n_features)
    - t_vec: numpy array or list of treatment values, shape (n_samples,)
    - seed: optional int, for reproducibility

    Returns:
    - y: numpy array of shape (n_samples,)
    - v1, v2, v3: weight vectors used
    """
    if seed is not None:
        np.random.seed(seed)

    X = x_df.to_numpy()
    t_vec = np.asarray(t_vec).flatten()

    if X.shape[0] != t_vec.shape[0]:
        raise ValueError(f"Mismatch: x has {X.shape[0]} rows but t has {t_vec.shape[0]} values")

    d = X.shape[1]

    def sample_and_normalize():
        v = np.random.randn(d)
        return v / np.linalg.norm(v)

    v1 = sample_and_normalize()
    v2 = sample_and_normalize()
    v3 = sample_and_normalize()

    dot1 = X @ v1       # v1^T x
    dot2 = X @ v2       # v2^T x
    dot3 = X @ v3       # v3^T x

    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(dot3 != 0, dot2 / dot3, 0.0)  # avoid division by zero

    y = 10 * (dot1 + np.sin(ratio * np.pi * t_vec))
    return y, v1, v2, v3


