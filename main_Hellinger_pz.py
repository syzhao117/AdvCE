#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
import warnings
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.decomposition import PCA  # noqa: F401
from sklearn.preprocessing import StandardScaler  # noqa: F401

# ---------------- Project modules ----------------
from src.networks import HL_Counterfactual_Net
from src.utils import (
    compute_mise,
    compute_pe_2,
    compute_pe_tcga,
    compute_mise_tcga,
    get_true_y_news,
    get_true_y_tcga
)

from hellinger import HellingerMIEstimator

# ---------------- Global config ----------------
warnings.filterwarnings("ignore", category=UserWarning)
# SEED = 0
# np.random.seed(SEED)
# torch.manual_seed(SEED)

# ---------------- Argparse ----------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train HL Counterfactual Net (CPU-only, Hellinger regularizer).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--beta", type=float, default=0.001, help="Weight for Hellinger term")
    # p.add_argument("--gamma", type=float, default=0.1, help="Weight for latent regularizer")
    p.add_argument("--attention", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--spline", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--num_epochs", type=int, default=2000)
    p.add_argument("--setting", choices=("news",  "tcga"), default="news")


    # Two-loop / stability
    p.add_argument("--k_critic", type=int, default=100, help="Inner steps for critic per outer step")
    p.add_argument("--k_shuffle", type=int, default=10, help="K shuffles to estimate product marginal")
    p.add_argument("--score_clip", type=float, default=6.0, help="Clamp critic scores to [-δ, δ]")

    # Save path
    p.add_argument("--save_path", type=str, default="models_cpu/HL_counterfactual_net.pth",
                   help="Model save path")
    return p.parse_args()

# ---------------- Data utils ----------------
def get_data_paths(setting: str) -> Tuple[Path, Path, Path]:
    if setting == "news":
        base = Path("data/news")
    else:  # tcga
        base = Path("data/TCGA")
    return base / "train.npy", base / "test.npy", base / "eval_test.npy"

def load_numpy_arrays(setting: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_path, test_path, eval_path = get_data_paths(setting)
    train = np.load(train_path)
    test = np.load(test_path)
    eval_test = np.load(eval_path)
    return train, test, eval_test

def split_arrays(data: np.ndarray, setting: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        t_arr = data[:, -2]
        x_arr = data[:, :-2]
        y_arr = data[:, -1]
    return t_arr, x_arr, y_arr.reshape(-1, 1)

# ---------------- Model helpers ----------------
def build_model(x_dim: int, t_dim: int, use_attention: bool, use_spline: bool) -> nn.Module:
    hidden_dim = 512
    z_dim = 16 if t_dim > 1 else 32
    t_dim_latent = 8
    return HL_Counterfactual_Net(
        x_dim=x_dim,
        t_dim_latent=8,
        z_dim=8,
        y_dim=1,
        t_input_dim=t_dim,
        hidden_dim=hidden_dim,
        hidden_dim_t=t_dim_latent,
        attn_dim=64,
        use_attention=use_attention,
        use_spline=use_spline,
    )

# ---------------- Train (two-loop) ----------------
def train(
    model: nn.Module,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    y: torch.Tensor,
    beta: float,
    num_epochs: int,
    hellinger_estimator: "HellingerMIEstimator",
    k_critic: int,
    k_shuffle: int,
    score_clip: float,
    save_path: str,
):
    device = torch.device("cpu")  # CPU-only

    for epoch in range(1, num_epochs + 1):
        model.train()
        z, t_latent, y_pred = model(x, t)

        # (1) outcome loss (MSE)
        loss_y = criterion(y_pred, y)

        # (2) latent regularizer
        # loss_reg = torch.norm(z, dim=0).sum()

        # ------ inner: update critic k_critic times ------
        hellinger_estimator.model.train()
        J_inner_sum = 0.0
        for _ in range(k_critic):

            J_step= hellinger_estimator.train_step(
                z.detach(), t_latent.detach(),
                k_shuffle=k_shuffle,
                score_clip=score_clip
            )
            J_inner_sum += float(J_step)
        J_inner_avg = J_inner_sum / k_critic
        # ------ outer: freeze critic, update main model ------
        hellinger_estimator.model.eval()
        for p in hellinger_estimator.model.parameters():
            p.requires_grad_(False)

        # 外层要把梯度回传到主模型：detach=False
        J_outer = hellinger_estimator.compute_J(
            z, t_latent,
            k_shuffle=k_shuffle,
            detach=False,
            score_clip=score_clip
        )
        loss_hl = -J_outer

        # (3) total loss
        loss = loss_y + beta * loss_hl

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        # unfreeze critic
        for p in hellinger_estimator.model.parameters():
            p.requires_grad_(True)

        if epoch == num_epochs:
            print(
                f"Epoch [{epoch}/{num_epochs}] — "
                f"loss: {loss.item():.4f} | "
                f"MSE: {loss_y.item():.4f} | "
                f"Hellinger(inner/outer J): {{J_inner_avg:.4f}}/{J_outer.item():.4f} | "
                # f"Reg: {loss_reg.item():.4f}"
            )
            Path(Path(save_path).parent).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to: {save_path}")





# ---------------- Main ----------------
def main() -> None:
    args = parse_args()

    # Load
    train_arr, test_arr, _ = load_numpy_arrays(args.setting)
    t_arr, x_arr, y_arr = split_arrays(train_arr, args.setting)
    t_test_arr, x_test_arr, y_test_arr = split_arrays(test_arr, args.setting)

    # Tensors (CPU, float32)
    device = torch.device("cpu")
    x = torch.tensor(x_arr, dtype=torch.float32, device=device)
    t = torch.tensor(t_arr, dtype=torch.float32, device=device)
    y = torch.tensor(y_arr, dtype=torch.float32, device=device)
    x_test = torch.tensor(x_test_arr, dtype=torch.float32, device=device)
    t_test = torch.tensor(t_test_arr, dtype=torch.float32, device=device)
    y_test = torch.tensor(y_test_arr, dtype=torch.float32, device=device)

    # News: scalar treatment → add dim
    if args.setting in ("news", "tcga"):
        t = t.unsqueeze(1)
        t_test = t_test.unsqueeze(1)

    # Optional sanity check (now safe on CPU)
    if args.setting == "news":
        y_true = get_true_y_news(
            t_test.detach().cpu().numpy(),
            x_test.detach().cpu().numpy(),
            noise_sd=0.2,
        )
        df = pd.DataFrame({"y_true": y_true.flatten(), "y_test": y_test.flatten()})
        print(df.head())

    if args.setting == "tcga":
        v1, v2, v3 = np.load("data/TCGA/v_vector.npy")
        y_true = get_true_y_tcga(
            t_test.detach().cpu().numpy(),
            x_test.detach().cpu().numpy(),
            v1=v1, v2=v2, v3=v3,
            noise_sd=0.0,  # eval 推荐无噪声
        )
        df = pd.DataFrame({"y_true": y_true.flatten(), "y_test": y_test.flatten()})
        print(df.head())

    # Model (CPU)
    model = build_model(
        x_dim=x.shape[1],
        t_dim=t.shape[1],
        use_attention=args.attention,
        use_spline=args.spline,
    ).to(device)

    # Optimizer / loss
    lr = 1e-3
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    # Auto-infer estimator input size from (z, t_latent)
    with torch.no_grad():
        z_tmp, t_tmp, _ = model(x[:2], t[:2])
    input_dim_est = (z_tmp.shape[1] + t_tmp.shape[1])

    # Estimator (CPU)
    hellinger_estimator = HellingerMIEstimator(
        input_dim=input_dim_est,
        hidden=64,
        lr=1e-3,
        weight_decay=0.0,
        device="cpu",
    )
    hellinger_estimator.model = hellinger_estimator.model.to(device)

    # Train
    train(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        x=x,
        t=t,
        y=y,
        beta=args.beta,
        num_epochs=args.num_epochs,
        hellinger_estimator=hellinger_estimator,
        k_critic=args.k_critic,
        k_shuffle=args.k_shuffle,
        score_clip=args.score_clip,
        save_path=args.save_path,
    )
    # ===== Evaluate (branch-specific) ====================================
    if args.setting == "tcga":
        v1, v2, v3 = np.load("data/TCGA/v_vector.npy")

        X_test_np = x_test.detach().cpu().numpy()


        device_eval = "cpu"
        mise = compute_mise_tcga(
             model=model.eval(),
             X_test=X_test_np, v1=v1, v2=v2, v3=v3,
             n_grid=65, device=device_eval,
             # chunk_elems=200_000,  # 如果 N*G 很大、显存/内存吃紧就打开
         )
        print("TCGA  MISE:", mise)
        pe = compute_pe_tcga(
            model=model.eval(),
            X_test=X_test_np, v1=v1, v2=v2, v3=v3,
            n_grid=65, device=device_eval,
            # chunk_elems=200_000,
        )
        print("TCGA    PE:", pe)




    else args.setting == "news":

        true_fn, src = (get_true_y_news, "data")
        mise = compute_mise(
            model=model.eval(),
            x_test=x_test, y_test=y_test,
            get_true_y_fn=true_fn, source=src,
            t_dim=t.shape[1], x_dim=x.shape[1],
            n_dosage_points=20, noise_sd=0.2,
        )
        print("MISE:", mise)

        pe = compute_pe_2(
            model=model.eval(),
            x_test=x_test,
            get_true_y_fn=true_fn, source=src,
            t_dim=t.shape[1], x_dim=x.shape[1],
            n_dosage_points=20, noise_sd=0.2,
        )
        print("PE:", pe)





if __name__ == "__main__":
    main()
