import torch
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.special import expit
import torch
import torch.nn as nn
from src.config import SEED
import os
from sklearn.decomposition import PCA
from sklearn.neighbors import BallTree
from sklearn.svm import SVC
import abc
from bisect import bisect_left


def compute_mise(
    model,
    x_test,
    y_test,
    get_true_y_fn,
    source: str,
    t_dim: int,
    x_dim: int,
    n_dosage_points: int = 20,
    noise_sd: float = 0.2
) -> float:
    """
    Compute the root mean integrated squared error (MISE) over the dose-response space.

    Parameters:
        model: Callable taking (x, t) → (latent, other, y_pred)
        x_test: np.ndarray or torch.Tensor, shape (N, x_dim)
        y_test: np.ndarray or torch.Tensor, shape (N,)
        get_true_y_fn: callable, ground-truth function for y(t, x)
        source: str, either "mimic/data" (2D dosage) or "news/data" (1D dosage)
        t_dim: int, dimension of treatment vector
        x_dim: int, dimension of covariates
        n_dosage_points: int, number of discrete dosage points to integrate over
        noise_sd: float, passed to get_true_y_news (ignored for mimic)

    Returns:
        float: root MISE value
    """
    criterion = nn.MSELoss()
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    N = x_test.shape[0]

    criterion = nn.MSELoss()

    if source == "mimic/data":
        dose_a_range = np.linspace(0.0, 1.0, n_dosage_points)
        dose_b_range = np.linspace(0.0, 1.0, n_dosage_points)
        dose_combinations = np.array([[a, b] for a in dose_a_range for b in dose_b_range])
        step_size = (dose_a_range[1] - dose_a_range[0]) * (dose_b_range[1] - dose_b_range[0])
    else:
        dose_range = np.linspace(0.0, 1.0, n_dosage_points)
        dose_combinations = dose_range[:, np.newaxis]  # shape (n_dosage_points, 1)
        step_size = dose_range[1] - dose_range[0]

    mise = 0.0

    for n in range(N):
        x_n = x_test[n].unsqueeze(0)  # shape (1, x_dim)
        integral_error = 0.0

        #print(f"Current: {n/N}")

        for dose_vec in dose_combinations:
            t_input = torch.tensor(dose_vec, dtype=torch.float32).unsqueeze(0)  # shape (1, t_dim)

            # Model prediction
            _, _, y_pred = model(x_n, t_input)

            # Ground-truth outcome
            if source == "mimic/data":
                y_true_val = get_true_y_fn(t_input, x_n, dim_treat=t_dim, dim_cov=x_dim)
            else:
                y_true_val = get_true_y_fn(
                    t_input.detach().cpu().numpy(),
                    x_n.detach().cpu().numpy(),
                    noise_sd=noise_sd
                )

            y_true = torch.tensor(y_true_val, dtype=torch.float32)

            #error = criterion(y_true, y_pred)#(y_true - y_pred.detach().cpu().numpy())**2
            error = criterion(y_true,y_pred)
            integral_error += error.item()

        mise += integral_error * step_size

    mise /= N
    return np.sqrt(mise)

#

#
def compute_mise_tcga(model, X_test, v1, v2, v3, n_grid=65, device=None, chunk_elems=None):
    """
    Root MISE on TCGA (1D dosage). Vectorized over all samples & grid.
    - model: forward(x, t) -> (_, _, ŷ) with ŷ shape (B,) or (B,1)
    - X_test: (N, d) numpy or torch
    - v1,v2,v3: (d,) numpy
    - n_grid: number of dosage points (65 for TCGA)
    - device: torch device; auto-detect if None
    - chunk_elems: if not None, split (N*G) into chunks of this size to save memory
    """
    # device = device or (next(model.parameters()).device if any(p.requires_grad for p in model.parameters()) else "cpu")
    model.eval()

    # --- tensors & grid ---
    X = torch.as_tensor(X_test, dtype=torch.float32, device=device)
    N, d = X.shape
    t_grid = torch.linspace(0.0, 1.0, n_grid, dtype=torch.float32, device=device)[:, None]   # (G,1)
    dt = (1.0/(n_grid-1)) if n_grid > 1 else 1.0

    # --- true curves (N,G) via vectorized formula ---
    v1_t = torch.as_tensor(v1, dtype=torch.float32, device=device)
    v2_t = torch.as_tensor(v2, dtype=torch.float32, device=device)
    v3_t = torch.as_tensor(v3, dtype=torch.float32, device=device)

    dot1 = (X @ v1_t).unsqueeze(1)                            # (N,1)
    dot2 = (X @ v2_t).unsqueeze(1)                            # (N,1)
    dot3 = (X @ v3_t).unsqueeze(1)                            # (N,1)
    T = t_grid.T                                              # (1,G)
    Y_true = 10.0 * (dot1 + 12.0*dot2*T - 12.0*dot3*(T**2))   # (N,G)

    # --- predicted curves (N,G) in one (or few) forwards ---
    # Build repeated inputs for all (x,t) pairs
    def predict_all():
        # shape (N*G, d) and (N*G, 1)
        X_rep = X.repeat_interleave(n_grid, dim=0)
        T_rep = t_grid.repeat(N, 1)                           # (N*G,1)

        if chunk_elems is None:   # single forward
            _, _, Y_pred = model(X_rep, T_rep)                # (N*G,) or (N*G,1)
            Y_pred = Y_pred.view(-1).reshape(N, n_grid)
            return Y_pred

        # chunked forward to control memory
        out = []
        for s in range(0, N*n_grid, chunk_elems):
            e = min(s + chunk_elems, N*n_grid)
            _, _, y = model(X_rep[s:e], T_rep[s:e])
            out.append(y.view(-1))
        Y_pred = torch.cat(out, dim=0).reshape(N, n_grid)
        return Y_pred

    Y_pred = predict_all()                                    # (N,G)

    # --- MISE: sqrt( mean_i ∫ (ŷ - y)^2 dt ) ---
    sq_err = (Y_pred - Y_true).pow(2)                         # (N,G)
    mise = torch.sqrt( (sq_err.sum(dim=1) * dt).mean() )      # scalar
    return float(mise.detach().cpu().numpy())

def compute_pe_tcga(model, X_test, v1, v2, v3, n_grid=65, device=None, chunk_elems=None):
    """
    Policy Error on TCGA (1D dosage), vectorized.
    PE = mean( y_true(t*) - y_true(t_hat) ), with argmax on the same grid.
    """
    # device = device or (next(model.parameters()).device if any(p.requires_grad for p in model.parameters()) else "cpu")
    model.eval()

    X = torch.as_tensor(X_test, dtype=torch.float32, device=device)
    N, d = X.shape
    t_grid = torch.linspace(0.0, 1.0, n_grid, dtype=torch.float32, device=device)[:, None]   # (G,1)

    # true curves (N,G)
    v1_t = torch.as_tensor(v1, dtype=torch.float32, device=device)
    v2_t = torch.as_tensor(v2, dtype=torch.float32, device=device)
    v3_t = torch.as_tensor(v3, dtype=torch.float32, device=device)
    dot1 = (X @ v1_t).unsqueeze(1)
    dot2 = (X @ v2_t).unsqueeze(1)
    dot3 = (X @ v3_t).unsqueeze(1)
    T = t_grid.T
    Y_true = 10.0 * (dot1 + 12.0*dot2*T - 12.0*dot3*(T**2))   # (N,G)

    # predicted curves (N,G): same trick as above
    def predict_all():
        X_rep = X.repeat_interleave(n_grid, dim=0)
        T_rep = t_grid.repeat(N, 1)
        if chunk_elems is None:
            _, _, Y_pred = model(X_rep, T_rep)
            return Y_pred.view(-1).reshape(N, n_grid)
        outs = []
        for s in range(0, N*n_grid, chunk_elems):
            e = min(s + chunk_elems, N*n_grid)
            _, _, y = model(X_rep[s:e], T_rep[s:e])
            outs.append(y.view(-1))
        return torch.cat(outs, dim=0).reshape(N, n_grid)

    Y_pred = predict_all()                                    # (N,G)

    # argmax on grid
    idx_true = torch.argmax(Y_true, dim=1)                    # (N,)
    idx_pred = torch.argmax(Y_pred, dim=1)                    # (N,)

    # gather y_true at those indices and average regret
    y_star = Y_true.gather(1, idx_true.unsqueeze(1)).squeeze(1)   # (N,)
    y_hat  = Y_true.gather(1, idx_pred.unsqueeze(1)).squeeze(1)   # (N,)
    pe = (y_star - y_hat)**2
    pe=pe.mean()
    return np.sqrt(pe.detach().cpu().numpy())

# def compute_mise_tcga(
#     model,
#     x_test,                         # (N, d)
#     get_true_y_tcga,                # 必须用固定的 v1,v2,v3（见下）
#     v1, v2, v3,                     # 固定好的向量
#     n_dosage_points: int = 65,      # TCGA：65
# ) -> float:
#     """
#     Root MISE for TCGA on an equally spaced 1D grid in [0,1].
#     get_true_y_tcga(t, x, v1,v2,v3) -> y_true (numpy or torch ok)
#     """
#     criterion = nn.MSELoss()
#     x_test = torch.as_tensor(x_test, dtype=torch.float32)
#     N = x_test.shape[0]
#
#     dose_range = np.linspace(0.0, 1.0, n_dosage_points, dtype=np.float32)
#     step_size  = dose_range[1] - dose_range[0] if n_dosage_points > 1 else 1.0
#
#     mise = 0.0
#
#     for n in range(N):
#         x_n = x_test[n:n+1]  # (1,d)
#         integral_error = 0.0
#
#         for t in dose_range:
#             t_input = torch.tensor([[t]], dtype=torch.float32)  # (1,1)
#
#             # model forward
#             _, _, y_pred = model(x_n, t_input)                  # shape (1,)
#             y_pred = y_pred.reshape(-1)                         # ensure (1,)
#
#             # ground truth (确保使用同一组 v1,v2,v3)
#             y_true_val = get_true_y_tcga(
#                 t_input.numpy(), x_n.numpy(), v1=v1, v2=v2, v3=v3, noise_sd=0.0
#             )
#             y_true = torch.as_tensor(y_true_val, dtype=torch.float32).reshape(-1)
#
#             integral_error += criterion(y_true, y_pred).item()
#
#         mise += integral_error * step_size
#
#     mise /= N
#     return float(np.sqrt(mise))

# def compute_pe_tcga(
#     model,
#     x_test,                 # (N, d)
#     v1, v2, v3,             # 固定向量
#     n_dosage_points: int = 65,
# ) -> float:
#     """
#     Policy Error (TCGA, 1D): mean( y_true(t*) - y_true(t_hat) )
#     t*  = argmax_t y_true(t),  t_hat = argmax_t y_pred(t)
#     使用固定 v1,v2,v3，在 [0,1] 的等距 65 点网格上离散求值。
#     """
#     x_test = torch.as_tensor(x_test, dtype=torch.float32)
#     N = x_test.shape[0]
#     t_grid = np.linspace(0.0, 1.0, n_dosage_points, dtype=np.float32)
#
#     regret_sum = 0.0
#     for i in range(N):
#         xi = x_test[i:i+1].numpy().astype(np.float32)      # (1,d)
#
#         # 真实曲线 y_true(t_k)
#         dot1 = float(xi @ v1); dot2 = float(xi @ v2); dot3 = float(xi @ v3)
#         y_true = 10.0 * (dot1 + 12.0*dot2*t_grid - 12.0*dot3*(t_grid**2))  # (G,)
#
#         # 模型曲线 y_pred(t_k)
#         with torch.no_grad():
#             t_tensor = torch.tensor(t_grid[:, None], dtype=torch.float32)    # (G,1)
#             x_rep    = torch.tensor(xi, dtype=torch.float32).repeat(len(t_grid), 1)  # (G,d)
#             _, _, y_pred = model(x_rep, t_tensor)                            # (G,1)
#             y_pred = y_pred.squeeze(-1).detach().cpu().numpy()               # (G,)
#
#         k_star = int(np.argmax(y_true))
#         k_hat  = int(np.argmax(y_pred))
#         regret_sum += (y_true[k_star] - y_true[k_hat])**2
#
#     return np.sqrt(regret_sum / N)

def get_true_y_tcga(t, x, v1, v2, v3, noise_sd=0.0):
    """
    x: (1,d) or (n,d), t: (1,1) or (n,1), v*: (d,)
    y = 10 * ( v1·x + 12*(v2·x)*t - 12*(v3·x)*t^2 )
    """
    x  = np.asarray(x, dtype=np.float32)
    tt = np.asarray(t, dtype=np.float32).reshape(-1)

    core = (x @ v1) + 12.0*(x @ v2)*tt - 12.0*(x @ v3)*(tt**2)
    y = 10.0 * core
    if noise_sd and noise_sd > 0:
        y = y + np.random.normal(0.0, noise_sd, size=y.shape).astype(np.float32)
    return y.astype(np.float32)

def get_true_y_mimic(t, x, param_interaction=2, dim_treat= 2, dim_cov = 10, noise=0.0):
    t = t.detach().numpy()
    x = x.detach().numpy()

    v = np.random.normal(loc=0.0, scale=1.0, size=(dim_treat, 2, dim_cov))
    v = v/(np.linalg.norm(v, 1, axis=2).reshape(dim_treat, 2, -1))
    
    pred_x = (np.float32(x[:, None,None,:]) * np.float32(v[None,...])).sum(3) # reducing float for big tcga dataset
    pred_x = pred_x[..., 0] / (2*pred_x[..., 1])
    pred_x = expit(pred_x)

    pred_x_adj = pred_x / 20 + 0.2

    y = 2 + 2 * (pred_x.mean(1)+0.5) * (np.cos((t - pred_x_adj)*3*np.pi) - 0.01*((t - pred_x_adj)**2)).mean(axis=1)\
        - param_interaction* 0.1*(((t-pred_x_adj)**2).prod(axis=1))
    noise = np.random.normal(0, noise, size=len(y))
    return y + noise




def compute_pe_2(
    model,
    x_test,
    get_true_y_fn,
    source: str,
    t_dim: int,
    x_dim: int,
    n_dosage_points: int = 20,
    noise_sd: float = 0.2,
) -> float:
    """
    Root Policy Error (RPE):  √(1/N · Σ (Y* - Y( t̂ ))²)

    Y*  = best-possible outcome on the dose grid (oracle)
    t̂  = dose the model would pick (argmax of its own prediction)
    """
    # ------------------------------------------------------------------ setup
    x_test = torch.as_tensor(x_test, dtype=torch.float32)
    N      = x_test.shape[0]

    # build the dose grid once ------------------------------------------------
    if source == "mimic/data":
        a = np.linspace(0.0, 1.0, n_dosage_points)
        b = np.linspace(0.0, 1.0, n_dosage_points)
        dose_combinations = np.array([[ai, bi] for ai in a for bi in b])  # (G, 2)
    else:  # "news/data"
        dose_combinations = np.linspace(0.0, 1.0, n_dosage_points)[:, None]  # (G, 1)

    t_grid = torch.as_tensor(dose_combinations, dtype=torch.float32)          # (G, t_dim)

    G      = t_grid.shape[0]

    # ------------------------------------------------------------------ loop
    sq_err_sum = 0.0
    sq_err_sum_me = 0.0

    with torch.no_grad():
        for x in x_test:                              # iterate over test points (1, x_dim)

            # replicate x to match grid size once per point -------------------
            x_rep = x.repeat(G, 1)                    # (G, x_dim)

            # model predictions on the whole grid ----------------------------
            _, _, y_pred = model(x_rep, t_grid)       # (G, 1)
            y_pred = y_pred.squeeze()                 # (G,)

            # true outcomes on the same grid ---------------------------------
            if source == "mimic/data":
                y_true = get_true_y_fn(t_grid, x_rep,
                                        dim_treat=t_dim, dim_cov=x_dim)       # (G,)
                y_true = torch.as_tensor(y_true, dtype=torch.float32)
            else:  # "news/data"
                y_true = get_true_y_fn(t_grid.cpu().numpy(),
                                        x.cpu().numpy(),
                                        noise_sd=noise_sd)                     # (G,)
                y_true = torch.as_tensor(y_true, dtype=torch.float32)

            # Retrieve the actual best true outcome and its index -------------------
            actual_best_dosage = y_true.argmax().item()  # index of the best true outcome
            outcome_under_best_actual = y_true[actual_best_dosage].item()  # true outcome at that index

            # Retrieve the predicted best true outcome and its index -------------------
            predicted_best_dosage = y_pred.argmax().item()
            outcome_under_best_predicted = y_true[predicted_best_dosage].item()    # true outcome at that index

            sq_err_sum_me += (outcome_under_best_actual - outcome_under_best_predicted) ** 2

            # oracle vs. policy ----------------------------------------------
            best_true_val  = y_true.max().item()            # oracle outcome
            best_pred_idx  = y_pred.argmax().item()         # model-optimal dose index
            y_true_at_pred = y_true[best_pred_idx].item()   # true outcome @ that dose

            sq_err_sum += (best_true_val - y_true_at_pred) ** 2

    return np.sqrt(sq_err_sum_me/ N)




class BatchAugmentation(object):
    def make_propensity_lists(self, train_ids, benchmark_implementation, **kwargs):
        match_on_covariates = kwargs["match_on_covariates"]
        if match_on_covariates:
            self.batch_augmentation = MahalanobisBatch()
        else:
            self.batch_augmentation = PropensityBatch()
        self.batch_augmentation.make_propensity_lists(train_ids, benchmark_implementation, **kwargs)

    def enhance_batch_with_propensity_matches(self, args, benchmark, treatment_data, input_data, batch_y,
                                              treatment_strengths, match_probability=1.0, num_randomised_neighbours=6):
        if self.batch_augmentation is not None:
            return self.batch_augmentation.enhance_batch_with_propensity_matches(args,
                                                                                 benchmark, treatment_data,
                                                                                 input_data, batch_y,
                                                                                 treatment_strengths,
                                                                                 match_probability,
                                                                                 num_randomised_neighbours)
        else:
            raise Exception("Batch augmentation mode must be set.")


# (N,)

def get_true_y_news(t: np.ndarray,
               x: np.ndarray,
               noise_sd: float = 0.2,
               eps: float = 1e-8) -> np.ndarray:
    """
    y = 10 · (v₁·x + sin(π · ratio · t)) + ε,    ε ~ 𝒩(0, noise_sd²)
    ratio = (v₂·x)/(v₃·x); ratio→0 if denominator tiny.
    """
    rng          = np.random.default_rng(SEED)
    try:
        v1p, v2p, v3p = rng.normal(0.0, 1.0, size=(3, x.shape[1]))
    except:
        v1p, v2p, v3p = rng.normal(0.0, 1.0, size=(3, x.shape[0]))
        
    v1, v2, v3    = [v / np.linalg.norm(v, 2) for v in (v1p, v2p, v3p)]
    v1, v2, v3 = v1.astype(np.float32), v2.astype(np.float32), v3.astype(np.float32)

    denom = x @ v3
    ratio = np.where(np.abs(denom) < eps, 0.0, (x @ v2) / denom)
    core  = (x @ v1) + np.sin(np.pi * ratio * t.squeeze())

    y     = 10.0 * core #+ np.random.normal(0.0, noise_sd, size=len(core))
    return y.astype(np.float32)

def rbf_kernel(x, y, sigma=1.0):
    """
    RBF kernel matrix: K[i, j] = exp(-||x_i - y_j||^2 / (2*sigma^2))
    """
    # x shape: (N, d), y shape: (M, d)
    # cdist => pairwise Eucl. distances => shape (N, M)
    dists = torch.cdist(x, y, p=2)**2  # squared distances
    K = torch.exp(-dists / (2 * sigma**2))
    return K

def mmd_loss(z, z_prior, sigma=1.0):
    """
    Compute MMD^2 between samples z and z_prior.
    MMD^2 = E[k(z,z)] + E[k(z_prior,z_prior)] - 2 E[k(z,z_prior)]
    """
    K_zz = rbf_kernel(z, z, sigma)
    K_pp = rbf_kernel(z_prior, z_prior, sigma)
    K_zp = rbf_kernel(z, z_prior, sigma)

    mmd = K_zz.mean() + K_pp.mean() - 2 * K_zp.mean()
    return mmd



def standardize_tensor_with_scaler(X_train, X_test=None):
    # Convert the PyTorch tensor to a NumPy array
    X_train_np = X_train.numpy()

    # Initialize the StandardScaler
    scaler = StandardScaler()

    # Fit the scaler on the training data and transform it
    X_train_standardized = scaler.fit_transform(X_train_np)

    # Convert back to a PyTorch tensor
    X_train_standardized_tensor = torch.tensor(X_train_standardized)

    if X_test is not None:
        # If test data is provided, apply the same transformation (without fitting again)
        X_test_np = X_test.numpy()
        X_test_standardized = scaler.transform(X_test_np)
        X_test_standardized_tensor = torch.tensor(X_test_standardized)

        return X_train_standardized_tensor, X_test_standardized_tensor

    return X_train_standardized_tensor
