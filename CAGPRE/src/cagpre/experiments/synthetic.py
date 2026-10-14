# -*- coding: utf-8 -*-
"""
CAGPRE Pipeline - SIMULATED Data (Aligned with Real-Data Residual Pipeline)
============================================================================
This version keeps the CAGPRE model, preprocessing, training objective, GQE,
and baseline configurations aligned with the real-data residual pipeline.

Intentional differences:
  1. The synthetic dataset is generated separately and loaded from CSV.
  2. The saved train/validation/test split is preserved exactly.
  3. Controlled synthetic errors are present in train, validation, and test,
     so validation tuning uses the same F1 + range-IoU selection criterion
     as the real-data pipeline.

Run generate_synthetic.py first. This script never falls back to the old
on-the-fly synthetic generator.
"""
import argparse
import gc
import math
import logging
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import optuna
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, QuantileTransformer, RobustScaler
from tqdm import tqdm

# Shared definitions from the real-world data infrastructure.
from ..engines import shared as _shared
set_seed = _shared.set_seed
Autoencoder = _shared.Autoencoder
VanillaVAE = _shared.VanillaVAE
TransformerAutoencoder = _shared.TransformerAutoencoder
train_autoencoder = _shared.train_autoencoder
evaluate_global_iqr = _shared.evaluate_global_iqr
compute_metrics = _shared.compute_metrics
compute_all_range_metrics = _shared.compute_all_range_metrics
save_model = _shared.save_model
save_results_csv = _shared.save_results_csv
_load_partial_results = _shared._load_partial_results
_save_partial_results = _shared._save_partial_results
_save_range_comparison = _shared._save_range_comparison

# ===========================================================================
# Dedicated Simulated Data Preprocessor
# ===========================================================================
class SimulatedTabularPreprocessor:
    """
    Handles encoding of categorical context + scaling of target and numeric features
    specifically optimized for SIMULATED data.
    Uses output_distribution="normal" so targets map to Gaussian N(0,1) space.
    Includes continuous age in num_cols for continuous linear age trends.
    """
    def __init__(self, cat_cols: List[str] = None, num_cols: List[str] = None, encoding_type: str = "onehot"):
        self.cat_cols = cat_cols or ["gender", "condition"]
        self.num_cols = num_cols if num_cols is not None else ["age"]
        self.encoding_type = encoding_type
        self.ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.ord = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        self.num_scaler = RobustScaler()
        self.target_scaler = QuantileTransformer(
            output_distribution="normal",
            n_quantiles=1000,
            random_state=42
        )
        self._fitted = False
        self.cat_dims = []

    def fit(self, df: pd.DataFrame) -> "SimulatedTabularPreprocessor":
        if self.encoding_type == "embedding":
            self.ord.fit(df[self.cat_cols])
            self.cat_dims = [len(c) for c in self.ord.categories_]
        else:
            self.ohe.fit(df[self.cat_cols])
        
        if self.num_cols:
            self.num_scaler.fit(df[self.num_cols].values)
        self.target_scaler.fit(df[["value"]].values)
        self._fitted = True
        return self

    def get_category_sizes(self) -> List[int]:
        return self.cat_dims

    def transform_context(self, df: pd.DataFrame) -> np.ndarray:
        num = np.zeros((len(df), 0), dtype=np.float32)
        if self.num_cols:
            num = self.num_scaler.transform(df[self.num_cols].values).astype(np.float32)
            
        if self.encoding_type == "embedding":
            cat = self.ord.transform(df[self.cat_cols])
            cat = np.clip(cat, 0, None).astype(np.float32)
            return np.hstack([num, cat])
        else:
            cat = self.ohe.transform(df[self.cat_cols]).astype(np.float32)
            return np.hstack([num, cat]) if num.shape[1] > 0 else cat

    def transform_target(self, values: np.ndarray) -> np.ndarray:
        return self.target_scaler.transform(values.reshape(-1, 1)).flatten().astype(np.float32)

    def inverse_target(self, scaled: np.ndarray) -> np.ndarray:
        return self.target_scaler.inverse_transform(scaled.reshape(-1, 1)).flatten()


TabularPreprocessor = SimulatedTabularPreprocessor

# ===========================================================================
# Simulated Data Preparation
# ===========================================================================
def prepare_datasets_simulated(
    df: pd.DataFrame,
    extra_context: List[str],
    seed: int,
    encoding_type: str = "onehot",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, TabularPreprocessor]:
    """
    Preserve the split written by generate_synthetic.py.

    Simulated-data preprocessing intentionally uses:
      categorical context: gender, condition
      numeric context:     continuous age + extra_context
      target scaler:       QuantileTransformer(output_distribution="normal",
                           n_quantiles=1000)

    This differs intentionally from the large real-data preprocessing because
    the synthetic experiment contains only 20,000 records and the Gaussian
    target transform is more stable for this setting.

    The synthetic data are NEVER randomly re-split here.
    """
    df = df.copy()

    if "condition" in df.columns:
        df["condition"] = df["condition"].fillna("None").astype(str)
    if "gender" in df.columns:
        df["gender"] = df["gender"].fillna("UNKNOWN").astype(str)

    if "age_group" not in df.columns:
        if "age" not in df.columns:
            raise ValueError("Synthetic dataset must contain 'age' or 'age_group'.")
        df["age_group"] = pd.cut(
            df["age"],
            bins=[0, 19, 29, 39, 49, 59, 69, 79, 89, 200],
            labels=["0-19", "20-29", "30-39", "40-49", "50-59",
                    "60-69", "70-79", "80-89", "90+"],
        ).astype(str)

    if "split" not in df.columns:
        raise ValueError(
            "Synthetic dataset has no 'split' column. Run generate_synthetic.py "
            "first and use the saved CSV; do not re-split the data here."
        )

    valid_splits = {"train", "val", "test"}
    observed_splits = set(df["split"].dropna().astype(str).unique())
    missing = valid_splits - observed_splits
    if missing:
        raise ValueError(f"Synthetic dataset is missing split(s): {sorted(missing)}")

    train_df = df[df["split"] == "train"].copy().reset_index(drop=True)
    val_df = df[df["split"] == "val"].copy().reset_index(drop=True)
    test_df = df[df["split"] == "test"].copy().reset_index(drop=True)

    print(
        f"  Fixed split: train={len(train_df):,}  "
        f"val={len(val_df):,}  test={len(test_df):,}"
    )
    print(
        f"  Errors: train={int(train_df['is_error'].sum()):,}  "
        f"val={int(val_df['is_error'].sum()):,}  "
        f"test={int(test_df['is_error'].sum()):,}"
    )

    cat_cols = ["gender", "condition"]
    num_cols = ["age"] + extra_context

    preprocessor = SimulatedTabularPreprocessor(
        cat_cols=cat_cols,
        num_cols=num_cols,
        encoding_type=encoding_type,
    )
    preprocessor.fit(train_df)

    return train_df, val_df, test_df, preprocessor


# ===========================================================================
# ResNet Model Architecture & Cyclical Annealing
# ===========================================================================
class ResidualBlock(nn.Module):
    """Standard Residual Block with LayerNorm for tabular stability."""
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class RobustCAGPRECVAE(nn.Module):
    """
    100% Unsupervised Context-Aware Generative Model.
    Optimized with LayerNorm and Residual Connections for tighter distributions.
    """
    def __init__(self, context_dim: int, hidden_dim: int, latent_dim: int,
                 encoding_type: str = "onehot", cat_dims: List[int] = None, embed_dim: int = 8,
                 num_layers: int = 2):
        super().__init__()
        self.encoding_type = encoding_type
        self.cat_dims = cat_dims or []
        self.embed_dim = embed_dim
        
        self.embeddings = nn.ModuleList()
        if encoding_type == "embedding" and self.cat_dims:
            for size in self.cat_dims:
                self.embeddings.append(nn.Embedding(size + 1, embed_dim))
            num_continuous = context_dim - len(self.cat_dims)
            actual_context_dim = num_continuous + (len(self.cat_dims) * embed_dim)
        else:
            actual_context_dim = context_dim
            
        # --- ENCODER ---
        in_dim = actual_context_dim + 1
        encoder_layers = [
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2)
        ]
        for _ in range(num_layers - 1):
            encoder_layers.append(ResidualBlock(hidden_dim, dropout=0.1))
            
        self.encoder_net = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # --- DECODER ---
        in_dim = latent_dim + actual_context_dim
        decoder_layers = [
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2)
        ]
        for _ in range(num_layers - 1):
            decoder_layers.append(ResidualBlock(hidden_dim, dropout=0.1))
            
        decoder_layers.append(nn.Linear(hidden_dim, 1))
        self.decoder_net = nn.Sequential(*decoder_layers)

    def _process_context(self, c: torch.Tensor) -> torch.Tensor:
        if self.encoding_type == "embedding" and self.cat_dims:
            num_cats = len(self.cat_dims)
            c_num = c[:, :-num_cats]
            c_cat = c[:, -num_cats:].long()
            emb_outs = [emb_layer(c_cat[:, i]) for i, emb_layer in enumerate(self.embeddings)]
            return torch.cat([c_num] + emb_outs, dim=1)
        return c

    def encode(self, x: torch.Tensor, c: torch.Tensor):
        inputs = torch.cat([x, self._process_context(c)], dim=1)
        h = self.encoder_net(inputs)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z, c):
        inputs = torch.cat([z, self._process_context(c)], dim=1)
        return self.decoder_net(inputs)

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar


def get_cyclical_beta(epoch: int, total_epochs: int, cycles: int = 4, max_beta: float = 1.0, ratio: float = 0.5) -> float:
    """Calculates the Cyclical KL annealing weight for a given epoch."""
    period = total_epochs / cycles
    step = (epoch % period) / period
    if step < ratio:
        return max_beta * (step / ratio)
    return max_beta


# ===========================================================================
# Training Logic - aligned with real-data residual pipeline
# ===========================================================================
def train_cagpre_residual(
    model: RobustCAGPRECVAE,
    x_train: np.ndarray,
    c_train: np.ndarray,
    x_val: np.ndarray,
    c_val: np.ndarray,
    epochs: int,
    lr: float,
    device: str,
    beta: float = 0.1,
    batch_size: Optional[int] = None,
    patience: int = 15,
    verbose: bool = True,
) -> None:
    """Use the same reconstruction-loss rule and cyclical KL schedule as real data."""
    if batch_size is None:
        batch_size = getattr(globals().get("args"), "batch_size", 1024)

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

    n_train = len(x_train)

    global_args = globals().get("args")
    variable = (
        global_args.variable
        if global_args is not None and hasattr(global_args, "variable")
        else "SBP"
    ) or "SBP"

    # EXACTLY match the real-data residual pipeline.
    if variable.upper() in ["GLUCOSE", "PR"]:
        robust_criterion = nn.MSELoss()
        criterion_name = "MSE"
    else:
        robust_criterion = nn.SmoothL1Loss(beta=1.0)
        criterion_name = "SmoothL1/Huber"

    if verbose:
        print(f"  Reconstruction loss for {variable.upper()}: {criterion_name}")

    x_train_t = torch.tensor(x_train, dtype=torch.float32, device=device).unsqueeze(1)
    c_train_t = torch.tensor(c_train, dtype=torch.float32, device=device)
    x_val_t = torch.tensor(x_val, dtype=torch.float32, device=device).unsqueeze(1)
    c_val_t = torch.tensor(c_val, dtype=torch.float32, device=device)

    use_amp = device != "cpu"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_state = None
    training_history = []

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)
        epoch_recon = 0.0
        epoch_kl = 0.0
        actual_batches = 0

        current_beta = get_cyclical_beta(
            epoch, epochs, cycles=4, max_beta=beta, ratio=0.5
        )

        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            batch_x = x_train_t[idx]
            batch_c = c_train_t[idx]

            with torch.amp.autocast(
                device_type=device if device != "cpu" else "cpu",
                enabled=use_amp,
            ):
                recon, mu, logvar = model(batch_x, batch_c)
                recon_loss = robust_criterion(recon, batch_x)
                kl_loss = -0.5 * torch.mean(
                    1 + logvar - mu.pow(2) - logvar.exp()
                )
                loss = recon_loss + current_beta * kl_loss

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_recon += recon_loss.item()
            epoch_kl += kl_loss.item()
            actual_batches += 1

        model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for start in range(0, len(x_val), batch_size):
                end = min(start + batch_size, len(x_val))
                x_vb = x_val_t[start:end]
                c_vb = c_val_t[start:end]

                with torch.amp.autocast(
                    device_type=device if device != "cpu" else "cpu",
                    enabled=use_amp,
                ):
                    recon, mu, logvar = model(x_vb, c_vb)
                    r_loss = robust_criterion(recon, x_vb)
                    k_loss = -0.5 * torch.mean(
                        1 + logvar - mu.pow(2) - logvar.exp()
                    )
                    v_loss = r_loss + beta * k_loss

                if not (torch.isnan(v_loss) or torch.isinf(v_loss)):
                    val_loss += v_loss.item()
                    val_batches += 1

        if val_batches > 0:
            avg_val_loss = val_loss / val_batches

            # Same safety lock as the real pipeline: only evaluate early
            # stopping during the flat/max-beta part of a cycle.
            if current_beta >= beta:
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    epochs_no_improve = 0
                    best_state = {
                        k: v.detach().cpu().clone()
                        for k, v in model.state_dict().items()
                    }
                else:
                    epochs_no_improve += 1

            scheduler.step(avg_val_loss)
        else:
            avg_val_loss = float("nan")

        denom = max(actual_batches, 1)
        training_history.append({
            "epoch": epoch,
            "beta": current_beta,
            "recon_loss": epoch_recon / denom,
            "kl_loss": epoch_kl / denom,
            "val_loss": avg_val_loss,
        })

        if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"    Epoch {epoch + 1:3d}/{epochs}  "
                f"recon={epoch_recon / denom:.4f}  "
                f"kl={epoch_kl / denom:.4f}  "
                f"val={avg_val_loss:.6f}  "
                f"beta={current_beta:.4f}  "
                f"lr={current_lr:.1e}  "
                f"patience={patience - epochs_no_improve}"
            )

        if epochs_no_improve >= patience:
            if verbose:
                print(
                    f"    Early stopping at epoch {epoch + 1} "
                    f"(no improvement for {patience} epochs)"
                )
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
        if verbose:
            print(f"    Restored best model (val_loss={best_val_loss:.4f})")

    try:
        save_dir = getattr(globals().get("args"), "save_dir", "artifacts_simulated")
        hist_dir = os.path.join(save_dir, variable.upper())
        os.makedirs(hist_dir, exist_ok=True)
        hist_path = os.path.join(hist_dir, f"training_history_{variable.upper()}.csv")
        pd.DataFrame(training_history).to_csv(hist_path, index=False)
        if verbose:
            print(f"    Saved training history to {hist_path}")
    except Exception as exc:
        print(f"    [WARN] Could not save training history: {exc}")


# ===========================================================================
# Range Generation (GQE) with Dictionary Caching & GPU Vectorized KDE
# ===========================================================================
def generate_ranges_residual(
    model: RobustCAGPRECVAE,
    context: np.ndarray,
    preprocessor: TabularPreprocessor,
    n_samples: int,
    alpha: float,
    device: str,
    batch_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Memory-safe GQE range generation.

    Important:
      - Contexts are processed in small batches.
      - Generated samples are written directly into one preallocated float32
        matrix instead of accumulating a list and concatenating it.
      - The Monte Carlo procedure and requested quantiles are unchanged.
    """
    model.eval()

    unique_contexts, inverse_indices = np.unique(
        context, axis=0, return_inverse=True
    )
    n_unique = unique_contexts.shape[0]

    print(
        f"  GQE: {n_unique:,} unique contexts, "
        f"{n_samples:,} samples/context, "
        f"context_batch={batch_size}"
    )

    unique_low_arr = np.empty(n_unique, dtype=np.float64)
    unique_high_arr = np.empty(n_unique, dtype=np.float64)
    unique_min_arr = np.empty(n_unique, dtype=np.float64)
    unique_max_arr = np.empty(n_unique, dtype=np.float64)

    latent_dim = model.fc_mu.out_features
    sample_chunk_size = 500

    with torch.no_grad():
        for start in tqdm(
            range(0, n_unique, batch_size),
            desc="GQE context batches",
            leave=True,
        ):
            end = min(start + batch_size, n_unique)
            chunk = unique_contexts[start:end]
            chunk_len = end - start

            c_t = torch.tensor(
                chunk, dtype=torch.float32, device=device
            )

            # At most 64 x n_samples values are retained at once.
            decoded = np.empty(
                (chunk_len, n_samples), dtype=np.float32
            )

            for s_start in range(
                0, n_samples, sample_chunk_size
            ):
                s_end = min(
                    s_start + sample_chunk_size, n_samples
                )
                s_count = s_end - s_start

                c_expanded = (
                    c_t.unsqueeze(1)
                    .expand(-1, s_count, -1)
                    .reshape(-1, c_t.shape[1])
                )

                z = torch.randn(
                    chunk_len * s_count,
                    latent_dim,
                    device=device,
                )

                generated = model.decode(
                    z, c_expanded
                ).view(chunk_len, s_count)

                # inverse_transform uses float64 internally. Keep the input
                # small, then immediately cast the result back to float32.
                gen_cpu = generated.detach().cpu().numpy()
                gen_real = preprocessor.inverse_target(
                    gen_cpu.reshape(-1)
                ).reshape(chunk_len, s_count)

                decoded[:, s_start:s_end] = gen_real.astype(
                    np.float32, copy=False
                )

                del c_expanded, z, generated, gen_cpu, gen_real

            unique_low_arr[start:end] = np.quantile(
                decoded, alpha / 2.0, axis=1
            )
            unique_high_arr[start:end] = np.quantile(
                decoded, 1.0 - alpha / 2.0, axis=1
            )
            unique_min_arr[start:end] = np.min(
                decoded, axis=1
            )
            unique_max_arr[start:end] = np.max(
                decoded, axis=1
            )

            del decoded, c_t
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    low_arr = unique_low_arr[inverse_indices]
    high_arr = unique_high_arr[inverse_indices]
    min_arr = unique_min_arr[inverse_indices]
    max_arr = unique_max_arr[inverse_indices]

    return low_arr, high_arr, min_arr, max_arr


def evaluate_residual_multi_alpha(
    model: RobustCAGPRECVAE,
    preprocessor: TabularPreprocessor,
    df: pd.DataFrame,
    n_samples: int,
    alphas: List[float],
    device: str,
    batch_size: int = 64,
    verbose: bool = True,
) -> Dict[float, Dict[str, float]]:
    """
    Evaluate several alpha values from ONE Monte Carlo GQE sample.

    The previous tuner regenerated 20k/35k samples separately for every alpha.
    That multiplied runtime and memory churn by the number of alpha values.
    Here each trial generates samples once and derives all requested quantiles
    from the same samples, which is both more efficient and a fairer alpha
    comparison.
    """
    model.eval()

    c = preprocessor.transform_context(df)
    values = df["value"].values
    y_true = df["is_error"].values

    unique_contexts, inverse_indices = np.unique(
        c, axis=0, return_inverse=True
    )
    n_unique = unique_contexts.shape[0]

    alphas = [float(a) for a in alphas]
    low_by_alpha = {
        a: np.empty(n_unique, dtype=np.float64)
        for a in alphas
    }
    high_by_alpha = {
        a: np.empty(n_unique, dtype=np.float64)
        for a in alphas
    }

    latent_dim = model.fc_mu.out_features
    sample_chunk_size = 500

    if verbose:
        print(
            f"  GQE tuning: {n_unique:,} unique contexts, "
            f"{n_samples:,} samples/context, "
            f"{len(alphas)} alpha values from one sample"
        )

    with torch.no_grad():
        for start in tqdm(
            range(0, n_unique, batch_size),
            desc="GQE tuning batches",
            leave=True,
            disable=not verbose,
        ):
            end = min(start + batch_size, n_unique)
            chunk = unique_contexts[start:end]
            chunk_len = end - start

            c_t = torch.tensor(
                chunk, dtype=torch.float32, device=device
            )

            decoded = np.empty(
                (chunk_len, n_samples), dtype=np.float32
            )

            for s_start in range(
                0, n_samples, sample_chunk_size
            ):
                s_end = min(
                    s_start + sample_chunk_size, n_samples
                )
                s_count = s_end - s_start

                c_expanded = (
                    c_t.unsqueeze(1)
                    .expand(-1, s_count, -1)
                    .reshape(-1, c_t.shape[1])
                )

                z = torch.randn(
                    chunk_len * s_count,
                    latent_dim,
                    device=device,
                )

                generated = model.decode(
                    z, c_expanded
                ).view(chunk_len, s_count)

                gen_cpu = generated.detach().cpu().numpy()
                gen_real = preprocessor.inverse_target(
                    gen_cpu.reshape(-1)
                ).reshape(chunk_len, s_count)

                decoded[:, s_start:s_end] = gen_real.astype(
                    np.float32, copy=False
                )

                del c_expanded, z, generated, gen_cpu, gen_real

            for a in alphas:
                low_by_alpha[a][start:end] = np.quantile(
                    decoded, a / 2.0, axis=1
                )
                high_by_alpha[a][start:end] = np.quantile(
                    decoded, 1.0 - a / 2.0, axis=1
                )

            del decoded, c_t
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    metrics_by_alpha: Dict[float, Dict[str, float]] = {}

    for a in alphas:
        pred_low = low_by_alpha[a][inverse_indices]
        pred_high = high_by_alpha[a][inverse_indices]

        y_pred = (
            (values < pred_low) |
            (values > pred_high)
        ).astype(int)

        metrics = compute_metrics(y_true, y_pred)
        metrics.update(
            compute_all_range_metrics(
                pred_low,
                pred_high,
                df["gt_low"].values,
                df["gt_high"].values,
                values,
            )
        )
        metrics_by_alpha[a] = metrics

    return metrics_by_alpha


def evaluate_residual(
    model: RobustCAGPRECVAE,
    preprocessor: TabularPreprocessor,
    df: pd.DataFrame,
    n_samples: int,
    alpha: float,
    device: str,
    return_ranges: bool = False,
) -> Union[Dict[str, float], Tuple]:
    c = preprocessor.transform_context(df)
    values = df["value"].values

    pred_low, pred_high, pred_min, pred_max = generate_ranges_residual(
        model, c, preprocessor, n_samples, alpha, device
    )

    y_pred = (values < pred_low) | (values > pred_high)
    metrics = compute_metrics(df["is_error"].values, y_pred.astype(int))
    metrics.update(compute_all_range_metrics(
        pred_low, pred_high, df["gt_low"].values, df["gt_high"].values, values
    ))

    if return_ranges:
        return metrics, pred_low, pred_high, pred_min, pred_max
    return metrics


# ===========================================================================
# Optuna Tuning - same search space as real pipeline
# ===========================================================================
def tune_residual(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    preprocessor: TabularPreprocessor,
    n_trials: int,
    default_alpha: float,
    default_n_samples: int,
    device: str,
    save_dir: str,
) -> Tuple[str, Dict[str, float]]:
    x_train = preprocessor.transform_target(train_df["value"].values)
    c_train = preprocessor.transform_context(train_df)
    x_val = preprocessor.transform_target(val_df["value"].values)
    c_val = preprocessor.transform_context(val_df)

    val_positive = int(val_df["is_error"].sum())
    val_negative = int(len(val_df) - val_positive)

    if val_positive == 0 or val_negative == 0:
        raise ValueError(
            "Validation tuning requires both plausible and implausible "
            "observations. Regenerate the synthetic data with the updated "
            "generate_synthetic.py."
        )


    best_overall_score = -1.0
    best_path = os.path.join(save_dir, "best_residual.pt")
    best_metrics: Dict[str, float] = {}

    global_args = globals().get("args")
    encoding_type = global_args.encoding_type if global_args else "onehot"
    embed_dim = global_args.embed_dim if global_args else 8
    variable = (
        global_args.variable
        if global_args is not None and hasattr(global_args, "variable")
        else "SBP"
    ) or "SBP"

    def objective(trial):
        nonlocal best_overall_score, best_metrics

        hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256])
        latent_dim = trial.suggest_categorical("latent_dim", [8, 16])
        num_layers = trial.suggest_int("num_layers", 2, 4)
        batch_size = trial.suggest_categorical("batch_size", [1024, 4096, 8192])
        lr = trial.suggest_categorical("lr", [1e-3, 5e-4, 2e-4])
        epochs = trial.suggest_categorical("epochs", [60, 80, 100, 120])

        if variable.upper() in ["GLUCOSE", "PR"]:
            beta = trial.suggest_categorical("beta", [0.01, 0.05, 0.1])
        else:
            beta = trial.suggest_categorical("beta", [0.1, 0.15, 0.2, 0.3, 0.5])

        # Same alpha candidate set for every variable, matching the
        # real-data tuning procedure.
        alphas_to_test = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10]

        trial_n_samples = trial.suggest_categorical("n_samples", [5000, 10000, 20000])

        model = RobustCAGPRECVAE(
            context_dim=c_train.shape[1],
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            encoding_type=encoding_type,
            cat_dims=preprocessor.get_category_sizes(),
            embed_dim=embed_dim,
            num_layers=num_layers,
        )

        train_cagpre_residual(
            model, x_train, c_train, x_val, c_val,
            epochs=epochs, lr=lr, device=device, beta=beta, batch_size=batch_size,
            verbose=False,
        )

        best_trial_score = -1.0
        best_trial_alpha = default_alpha
        best_trial_metrics = {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "iou": 0.0,
        }

        # Generate the Monte Carlo sample ONCE for this trial and evaluate
        # all alpha candidates from the same generated values.
        metrics_by_alpha = evaluate_residual_multi_alpha(
            model=model,
            preprocessor=preprocessor,
            df=val_df,
            n_samples=trial_n_samples,
            alphas=alphas_to_test,
            device=device,
            verbose=False,
        )

        for test_alpha in alphas_to_test:
            metrics = metrics_by_alpha[float(test_alpha)]

            prec_val = metrics.get("precision", 0.0)
            rec_val = metrics.get("recall", 0.0)
            f1_val = metrics.get("f1", 0.0)
            iou_val = metrics.get("iou", 0.0)

            # Same selection criterion as the real-data pipeline:
            # harmonic mean of validation F1 and validation range IoU.
            if f1_val > 0.0 and iou_val > 0.0:
                current_score = (
                    2.0 * f1_val * iou_val
                    / (f1_val + iou_val)
                )
            else:
                current_score = 0.0

            if current_score >= best_trial_score:
                best_trial_score = current_score
                best_trial_alpha = test_alpha
                best_trial_metrics = metrics

        del metrics_by_alpha
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        trial.set_user_attr("precision", best_trial_metrics.get("precision", 0.0))
        trial.set_user_attr("recall", best_trial_metrics.get("recall", 0.0))
        trial.set_user_attr("f1", best_trial_metrics.get("f1", 0.0))
        trial.set_user_attr("iou", best_trial_metrics.get("iou", 0.0))
        trial.set_user_attr("alpha", best_trial_alpha)

        print(
            f"\n--- Trial {trial.number} ---\n"
            f"score={best_trial_score:.4f}  alpha={best_trial_alpha}  "
            f"hidden={hidden_dim}  latent={latent_dim}  layers={num_layers}  "
            f"beta={beta}  n_samples={trial_n_samples}\n"
            f"P={best_trial_metrics.get('precision', 0.0):.4f}  "
            f"R={best_trial_metrics.get('recall', 0.0):.4f}  "
            f"F1={best_trial_metrics.get('f1', 0.0):.4f}  "
            f"IoU={best_trial_metrics.get('iou', 0.0):.4f}"
        )

        if best_trial_score > best_overall_score:
            best_overall_score = best_trial_score
            best_metrics = best_trial_metrics
            meta = {
                "hidden_dim": hidden_dim,
                "latent_dim": latent_dim,
                "num_layers": num_layers,
                "batch_size": batch_size,
                "lr": lr,
                "epochs": epochs,
                "beta": beta,
                "alpha": best_trial_alpha,
                "n_samples": trial_n_samples,
                "model_type": "residual_simulated_aligned",
            }
            save_model(model, best_path, meta)
            print("*** NEW BEST MODEL FOUND ***")

        return best_trial_score

    class _SuppressStudyCreation(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return not record.getMessage().startswith("A new study created")

    optuna_logger = optuna.logging.get_logger("optuna")
    creation_filter = _SuppressStudyCreation()
    # Optuna uses separate named loggers for study creation and trial results.
    # Attach to handlers so only study-creation messages are suppressed.
    for handler in optuna_logger.handlers:
        handler.addFilter(creation_filter)
    try:
        study = optuna.create_study(direction="maximize")
    finally:
        for handler in optuna_logger.handlers:
            handler.removeFilter(creation_filter)
    study.optimize(objective, n_trials=n_trials, catch=(MemoryError,))

    os.makedirs(save_dir, exist_ok=True)
    trials_path = os.path.join(save_dir, "optuna_trials_residual_simulated.csv")
    try:
        study.trials_dataframe().to_csv(trials_path, index=False)
    except Exception as exc:
        print(f"[WARN] Could not save Optuna trials: {exc}")

    return best_path, best_metrics




def train_conditional_model(
    model: nn.Module,
    c_train: np.ndarray, x_train: np.ndarray,
    c_val: np.ndarray, x_val: np.ndarray,
    epochs: int, lr: float, device: str,
    is_cvae: bool = False,
    batch_size: int = 1024,
) -> None:
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6
    )
    criterion = nn.SmoothL1Loss(beta=1.0)
    use_amp = (device != "cpu")
    scaler = torch.amp.GradScaler(enabled=use_amp)
    n_train = len(x_train)

    best_loss = float("inf")
    best_state = None
    patience_counter = 0
    patience_limit = 20

    for epoch in range(epochs):
        model.train()
        perm = np.random.permutation(n_train)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            c_b = torch.tensor(c_train[idx], dtype=torch.float32, device=device)
            x_b = torch.tensor(x_train[idx], dtype=torch.float32, device=device).unsqueeze(1)

            with torch.amp.autocast(device_type=device if device != "cpu" else "cpu", enabled=use_amp):
                if is_cvae:
                    recon, mu, logvar = model(c_b, x_b)
                    recon_loss = criterion(recon, x_b)
                    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                    current_beta = min(0.1, 0.1 * (epoch + 1) / max(epochs // 2, 1))
                    loss = recon_loss + current_beta * kl_loss
                else:
                    pred = model(c_b)
                    loss = criterion(pred, x_b)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= patience_limit:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)




# ===========================================================================
# Value Scanning Ranges for Baseline Models
# ===========================================================================
def value_scan_ranges(
    model,
    preprocessor: TabularPreprocessor,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    device: str,
    model_type: str = "sklearn",
    n_candidates: int = 5000,
) -> Tuple[np.ndarray, np.ndarray]:
    test_copy = test_df.copy()
    test_copy["_age_group"] = pd.cut(
        test_copy["age"], bins=[0, 30, 40, 50, 60, 70, 80, 200],
        labels=["18-30", "31-40", "41-50", "51-60", "61-70", "71-80", "80+"]
    )
    train_copy = train_df.copy()
    train_copy["_age_group"] = pd.cut(
        train_copy["age"], bins=[0, 30, 40, 50, 60, 70, 80, 200],
        labels=["18-30", "31-40", "41-50", "51-60", "61-70", "71-80", "80+"]
    )

    group_cols = ["_age_group", "gender", "condition"]

    train_vals = train_df["value"].values
    v_min = np.percentile(train_vals, 0.1)
    v_max = np.percentile(train_vals, 99.9)
    padding = (v_max - v_min) * 0.2
    candidate_values = np.linspace(v_min - padding, v_max + padding, n_candidates)

    group_ranges = {}
    unique_groups = test_copy.groupby(group_cols, observed=True).size().reset_index()[group_cols]

    for _, grp in tqdm(
        unique_groups.iterrows(),
        total=len(unique_groups),
        desc=f"Value scanning ({model_type})",
        leave=True,
    ):
        key = tuple(grp[c] for c in group_cols)

        matching = train_copy
        for c, val in zip(group_cols, key):
            matching = matching[matching[c] == val]
        if len(matching) == 0:
            matching = train_copy
        rep_row = matching.iloc[0:1]

        cand_df = pd.DataFrame({
            "value": candidate_values,
            "age": [rep_row["age"].values[0]] * n_candidates,
            "gender": [rep_row["gender"].values[0]] * n_candidates,
            "condition": [rep_row["condition"].values[0]] * n_candidates,
        })
        for col in train_df.columns:
            if col not in cand_df.columns and col not in ["is_error", "gt_low", "gt_high"]:
                cand_df[col] = rep_row[col].values[0]

        c_cand = preprocessor.transform_context(cand_df)
        x_cand = preprocessor.transform_target(candidate_values)
        full_cand = np.hstack([c_cand, x_cand.reshape(-1, 1)])

        if model_type == "sklearn":
            scores = model.decision_function(full_cand)
            plausible_mask = scores >= 0.0
            plausible_vals = candidate_values[plausible_mask]
        elif model_type == "ae":
            model_nn = model
            model_nn.eval()
            x_train_full = np.hstack([
                preprocessor.transform_context(train_df),
                preprocessor.transform_target(train_df["value"].values).reshape(-1, 1)
            ])
            with torch.no_grad():
                train_errors = []
                for chunk_start in range(0, min(10000, len(x_train_full)), 2048):
                    chunk_end = min(chunk_start + 2048, len(x_train_full))
                    x_tr = torch.tensor(x_train_full[chunk_start:chunk_end], dtype=torch.float32, device=device)
                    out_tr = model_nn(x_tr)
                    recon_tr = out_tr[0] if isinstance(out_tr, tuple) else out_tr
                    train_errors.append(((recon_tr - x_tr) ** 2).mean(dim=1).cpu().numpy())
                train_errors = np.concatenate(train_errors)
            threshold = np.percentile(train_errors, 95)

            with torch.no_grad():
                x_cand_t = torch.tensor(full_cand, dtype=torch.float32, device=device)
                out_cand = model_nn(x_cand_t)
                recon_cand = out_cand[0] if isinstance(out_cand, tuple) else out_cand
                cand_errors = ((recon_cand - x_cand_t) ** 2).mean(dim=1).cpu().numpy()

            plausible_mask = cand_errors <= threshold
            plausible_vals = candidate_values[plausible_mask]
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        if len(plausible_vals) == 0:
            grp_vals = matching["value"].values if len(matching) > 10 else train_vals
            group_ranges[key] = (np.percentile(grp_vals, 2.5), np.percentile(grp_vals, 97.5))
        else:
            group_ranges[key] = (plausible_vals.min(), plausible_vals.max())

    pred_low = np.empty(len(test_df))
    pred_high = np.empty(len(test_df))

    fallback_lo = np.percentile(train_vals, 2.5)
    fallback_hi = np.percentile(train_vals, 97.5)

    test_keys = list(zip(
        test_copy["_age_group"],
        test_copy["gender"],
        test_copy["condition"]
    ))
    for i, key in enumerate(test_keys):
        lo, hi = group_ranges.get(key, (fallback_lo, fallback_hi))
        pred_low[i] = lo
        pred_high[i] = hi

    return pred_low, pred_high


# ===========================================================================
# Run Comparison for All Models on Simulated Data
# ===========================================================================
def run_compare_residual_simulated(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    preprocessor: TabularPreprocessor,
    alpha: float,
    n_samples: int,
    device: str,
    seed: int,
    csv_path: str = "",
    tuned_model_dir: str = "",
    hyperparams: Optional[Dict[str, Any]] = None,
    force_train: bool = False,
) -> Dict[str, Dict[str, float]]:
    if force_train:
        # A forced run must not reuse metrics produced from an older synthetic
        # dataset or an older model. This is especially important after
        # regenerating synthetic_<VARIABLE>.csv.
        results = {}
        if csv_path and os.path.exists(csv_path):
            print(
                f"  [FRESH COMPARE] Ignoring existing partial results: "
                f"{csv_path}"
            )
    else:
        results = _load_partial_results(csv_path) if csv_path else {}

    x_train = preprocessor.transform_target(train_df["value"].values)
    c_train = preprocessor.transform_context(train_df)
    x_val = preprocessor.transform_target(val_df["value"].values)
    c_val = preprocessor.transform_context(val_df)
    x_test = preprocessor.transform_target(test_df["value"].values)
    c_test = preprocessor.transform_context(test_df)

    hp = hyperparams or {}
    h_dim = hp.get("hidden_dim", 256)
    l_dim = hp.get("latent_dim", 16)
    n_layers = hp.get("num_layers", getattr(args, "num_layers", 2))
    lr = hp.get("lr", 5e-4)
    epochs = hp.get("epochs", 80)
    beta = hp.get("beta", 0.1)
    eval_alpha = hp.get("alpha", alpha)
    eval_n_samples = hp.get("n_samples", n_samples)
    batch_size = hp.get("batch_size", getattr(args, "batch_size", 1024))

    # --- Residual CAGPRE ---
    residual_low = residual_high = residual_min = residual_max = None
    if "CAGPRE_Residual" not in results:
        t0 = time.time()
        tuned_path = (
            os.path.join(tuned_model_dir, "best_residual.pt")
            if tuned_model_dir else ""
        )

        if tuned_path and os.path.exists(tuned_path) and not force_train:
            print(f"Loading tuned CAGPRE from {tuned_path} ...")
            checkpoint = torch.load(tuned_path, map_location=device, weights_only=False)
            meta = checkpoint.get("meta", {})
            h_dim = meta.get("hidden_dim", h_dim)
            l_dim = meta.get("latent_dim", l_dim)
            n_layers = meta.get("num_layers", n_layers)
            eval_alpha = meta.get("alpha", eval_alpha)
            eval_n_samples = meta.get("n_samples", eval_n_samples)
            batch_size = meta.get("batch_size", batch_size)

            print(
                "  Loaded checkpoint settings: "
                f"alpha={eval_alpha}, n_samples={eval_n_samples}, "
                f"hidden_dim={h_dim}, latent_dim={l_dim}, "
                f"num_layers={n_layers}, batch_size={batch_size}"
            )

            cagpre = RobustCAGPRECVAE(
                context_dim=c_train.shape[1],
                hidden_dim=h_dim,
                latent_dim=l_dim,
                encoding_type=getattr(args, "encoding_type", "onehot"),
                cat_dims=preprocessor.get_category_sizes(),
                embed_dim=getattr(args, "embed_dim", 8),
                num_layers=n_layers,
            )
            cagpre.load_state_dict(checkpoint["state_dict"])
            cagpre.to(device)
            t1 = time.time()
        else:
            print(
                "Training CAGPRE on SIMULATED data with the same residual "
                "architecture/training rule used for real data ..."
            )
            cagpre = RobustCAGPRECVAE(
                context_dim=c_train.shape[1],
                hidden_dim=h_dim,
                latent_dim=l_dim,
                encoding_type=getattr(args, "encoding_type", "onehot"),
                cat_dims=preprocessor.get_category_sizes(),
                embed_dim=getattr(args, "embed_dim", 8),
                num_layers=n_layers,
            )
            train_cagpre_residual(
                cagpre, x_train, c_train, x_val, c_val,
                epochs=epochs, lr=lr, device=device, beta=beta, batch_size=batch_size,
            )
            t1 = time.time()

        (
            results["CAGPRE_Residual"],
            residual_low, residual_high, residual_min, residual_max,
        ) = evaluate_residual(
            cagpre, preprocessor, test_df, eval_n_samples, eval_alpha, device,
            return_ranges=True,
        )

        results["CAGPRE_Residual"]["train_time"] = round(t1 - t0, 2)
        results["CAGPRE_Residual"]["test_time"] = round(time.time() - t1, 2)
        results["CAGPRE_Residual"]["time_sec"] = round(time.time() - t0, 2)
        print(
            f"  CAGPRE_Residual done ({results['CAGPRE_Residual']['time_sec']}s): "
            f"{results['CAGPRE_Residual']}"
        )
        if csv_path:
            _save_partial_results(results, csv_path)

    # --- Global IQR ---
    iqr_low_arr = iqr_high_arr = None
    if "Global_IQR" not in results:
        print("Evaluating Global IQR ...")
        t0 = time.time()
        results["Global_IQR"], iqr_low_arr, iqr_high_arr = evaluate_global_iqr(train_df, test_df, return_ranges=True)
        results["Global_IQR"]["time_sec"] = round(time.time() - t0, 2)
        print(f"  Global IQR done: {results['Global_IQR']}")
        if csv_path:
            _save_partial_results(results, csv_path)

    full_train = np.hstack([c_train, x_train.reshape(-1, 1)])
    full_val = np.hstack([c_val, x_val.reshape(-1, 1)])
    full_test = np.hstack([c_test, x_test.reshape(-1, 1)])

    MAX_SKLEARN_ROWS = 50_000
    if full_train.shape[0] > MAX_SKLEARN_ROWS:
        rng = np.random.default_rng(seed)
        sk_idx = rng.choice(full_train.shape[0], MAX_SKLEARN_ROWS, replace=False)
        sk_train = full_train[sk_idx]
    else:
        sk_train = full_train

    model_ranges = {}

    print(
        "  Baseline value-scanning evaluates many candidate values per "
        "context. Progress bars below show that the process is still active."
    )

    # --- Isolation Forest ---
    if "Isolation_Forest" not in results:
        from sklearn.ensemble import IsolationForest as IF
        print("Training Isolation Forest (n_estimators=100, value-scanning) ...")
        t0 = time.time()
        iso = IF(n_estimators=100, contamination=0.05, random_state=seed, max_features=1.0, bootstrap=True)
        iso.fit(sk_train)
        t1 = time.time()
        if_low, if_high = value_scan_ranges(iso, preprocessor, train_df, test_df, device, model_type="sklearn")
        values = test_df["value"].values
        y_pred_if = ((values < if_low) | (values > if_high)).astype(int)
        results["Isolation_Forest"] = compute_metrics(test_df["is_error"].values, y_pred_if)
        results["Isolation_Forest"].update(compute_all_range_metrics(
            if_low, if_high, test_df["gt_low"].values, test_df["gt_high"].values, values
        ))
        t2 = time.time()
        results["Isolation_Forest"]["train_time"] = round(t1 - t0, 2)
        results["Isolation_Forest"]["test_time"] = round(t2 - t1, 2)
        results["Isolation_Forest"]["time_sec"] = round(t2 - t0, 2)
        print(f"  Isolation Forest done ({results['Isolation_Forest']['time_sec']}s): {results['Isolation_Forest']}")
        model_ranges["Isolation_Forest"] = (if_low, if_high)
        if csv_path:
            _save_partial_results(results, csv_path)

    # --- LOF ---
    if "LOF" not in results:
        from sklearn.neighbors import LocalOutlierFactor as LOFModel
        print("Training LOF (n_neighbors=20, value-scanning) ...")
        t0 = time.time()
        lof = LOFModel(n_neighbors=20, novelty=True, contamination=0.05, metric="manhattan")
        lof.fit(sk_train)
        t1 = time.time()
        lof_low, lof_high = value_scan_ranges(lof, preprocessor, train_df, test_df, device, model_type="sklearn")
        values = test_df["value"].values
        y_pred_lof = ((values < lof_low) | (values > lof_high)).astype(int)
        results["LOF"] = compute_metrics(test_df["is_error"].values, y_pred_lof)
        results["LOF"].update(compute_all_range_metrics(
            lof_low, lof_high, test_df["gt_low"].values, test_df["gt_high"].values, values
        ))
        t2 = time.time()
        results["LOF"]["train_time"] = round(t1 - t0, 2)
        results["LOF"]["test_time"] = round(t2 - t1, 2)
        results["LOF"]["time_sec"] = round(t2 - t0, 2)
        print(f"  LOF done ({results['LOF']['time_sec']}s): {results['LOF']}")
        model_ranges["LOF"] = (lof_low, lof_high)
        if csv_path:
            _save_partial_results(results, csv_path)

    # --- Standard Autoencoder ---
    if "Autoencoder" not in results:
        print("Training Standard Autoencoder ...")
        t0 = time.time()
        ae = Autoencoder(input_dim=full_train.shape[1], hidden_dim=128)
        train_autoencoder(ae, full_train, full_val, 50, 5e-4, device, batch_size=4096)
        t1 = time.time()
        ae_low, ae_high = value_scan_ranges(ae, preprocessor, train_df, test_df, device, model_type="ae")
        values = test_df["value"].values
        y_pred_ae = ((values < ae_low) | (values > ae_high)).astype(int)
        results["Autoencoder"] = compute_metrics(test_df["is_error"].values, y_pred_ae)
        results["Autoencoder"].update(compute_all_range_metrics(
            ae_low, ae_high, test_df["gt_low"].values, test_df["gt_high"].values, values
        ))
        t2 = time.time()
        results["Autoencoder"]["train_time"] = round(t1 - t0, 2)
        results["Autoencoder"]["test_time"] = round(t2 - t1, 2)
        results["Autoencoder"]["time_sec"] = round(t2 - t0, 2)
        print(f"  Autoencoder done ({results['Autoencoder']['time_sec']}s): {results['Autoencoder']}")
        model_ranges["Autoencoder"] = (ae_low, ae_high)
        if csv_path:
            _save_partial_results(results, csv_path)

    # --- Vanilla VAE ---
    if "VAE" not in results:
        print("Training Vanilla VAE ...")
        t0 = time.time()
        vae = VanillaVAE(input_dim=full_train.shape[1], hidden_dim=128, latent_dim=16)
        train_autoencoder(vae, full_train, full_val, 50, 5e-4, device, is_vae=True, batch_size=4096)
        t1 = time.time()
        vae_low, vae_high = value_scan_ranges(vae, preprocessor, train_df, test_df, device, model_type="ae")
        values = test_df["value"].values
        y_pred_vae = ((values < vae_low) | (values > vae_high)).astype(int)
        results["VAE"] = compute_metrics(test_df["is_error"].values, y_pred_vae)
        results["VAE"].update(compute_all_range_metrics(
            vae_low, vae_high, test_df["gt_low"].values, test_df["gt_high"].values, values
        ))
        t2 = time.time()
        results["VAE"]["train_time"] = round(t1 - t0, 2)
        results["VAE"]["test_time"] = round(t2 - t1, 2)
        results["VAE"]["time_sec"] = round(t2 - t0, 2)
        print(f"  VAE done ({results['VAE']['time_sec']}s): {results['VAE']}")
        model_ranges["VAE"] = (vae_low, vae_high)
        if csv_path:
            _save_partial_results(results, csv_path)

    # --- Transformer AE ---
    if "Transformer_AE" not in results:
        print("Training Transformer Autoencoder ...")
        t0 = time.time()
        tae = TransformerAutoencoder(input_dim=full_train.shape[1], d_model=32, n_heads=4, n_layers=2)
        train_autoencoder(tae, full_train, full_val, 50, 5e-4, device)
        t1 = time.time()
        tae_low, tae_high = value_scan_ranges(tae, preprocessor, train_df, test_df, device, model_type="ae")
        values = test_df["value"].values
        y_pred_tae = ((values < tae_low) | (values > tae_high)).astype(int)
        results["Transformer_AE"] = compute_metrics(test_df["is_error"].values, y_pred_tae)
        results["Transformer_AE"].update(compute_all_range_metrics(
            tae_low, tae_high, test_df["gt_low"].values, test_df["gt_high"].values, values
        ))
        t2 = time.time()
        results["Transformer_AE"]["train_time"] = round(t1 - t0, 2)
        results["Transformer_AE"]["test_time"] = round(t2 - t1, 2)
        results["Transformer_AE"]["time_sec"] = round(t2 - t0, 2)
        print(f"  Transformer AE done ({results['Transformer_AE']['time_sec']}s): {results['Transformer_AE']}")
        model_ranges["Transformer_AE"] = (tae_low, tae_high)
        if csv_path:
            _save_partial_results(results, csv_path)

    # Save the same per-record range-comparison output used by the real pipeline.
    if csv_path and (residual_low is not None or iqr_low_arr is not None or model_ranges):
        range_csv = csv_path.replace("compare_", "range_comparison_")
        _save_range_comparison(
            test_df, residual_low, residual_high, None,
            iqr_low_arr, iqr_high_arr, range_csv,
            model_ranges=model_ranges,
        )

    return results


# ===========================================================================
# Main CLI Entry Point
# ===========================================================================
args = None


def _build_model_from_args(preprocessor: TabularPreprocessor, context_dim: int) -> RobustCAGPRECVAE:
    return RobustCAGPRECVAE(
        context_dim=context_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        encoding_type=args.encoding_type,
        cat_dims=preprocessor.get_category_sizes(),
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
    )


def main() -> None:
    global args

    parser = argparse.ArgumentParser(
        description=(
            "CAGPRE simulated-data pipeline aligned with the real-data "
            "residual CAGPRE pipeline"
        )
    )
    parser.add_argument(
        "--mode",
        choices=["tune", "train", "evaluate", "compare", "preview"],
        default="compare",
    )
    parser.add_argument("--dataset", choices=["simulated"], default="simulated")
    parser.add_argument("--variable", default="SBP", help="SBP, DBP, PR, CH, Glucose")
    parser.add_argument(
        "--synthetic-dir",
        default="synthetic_data",
        help="Directory containing synthetic_<VARIABLE>.csv files",
    )
    parser.add_argument("--save-dir", default="artifacts_simulated")
    parser.add_argument("--load-path", default=None)

    # Same model defaults as the real residual pipeline.
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.005)
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--encoding-type",
        choices=["onehot", "embedding"],
        default="onehot",
    )
    parser.add_argument("--embed-dim", type=int, default=8)
    parser.add_argument(
        "--force-train",
        action="store_true",
        help=(
            "Ignore an existing tuned checkpoint, retrain CAGPRE, and in "
            "compare mode ignore previously saved partial comparison results."
        ),
    )

    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.mode != "tune":
        print(f"Using device: {device}")

    variable = args.variable.upper()
    if variable not in {"SBP", "DBP", "PR", "CH", "GLUCOSE"}:
        raise ValueError("--variable must be one of SBP, DBP, PR, CH, GLUCOSE")

    synth_file = os.path.join(args.synthetic_dir, f"synthetic_{variable}.csv")
    if not os.path.exists(synth_file):
        raise FileNotFoundError(
            f"Synthetic dataset not found: {synth_file}\n"
            "Run generate_synthetic.py first. The aligned pipeline deliberately "
            "does not fall back to the old synthetic generator."
        )

    print(f"Loading fixed synthetic dataset from: {synth_file}")
    df = pd.read_csv(synth_file)

    required_columns = {
        "value", "age", "gender", "condition", "gt_low", "gt_high",
        "is_error", "split",
    }
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Synthetic CSV is missing required columns: {sorted(missing_columns)}"
        )

    print(
        f"Dataset: simulated | Variable: {variable} | Records: {len(df):,} | "
        f"Errors: {int(df['is_error'].sum()):,}"
    )

    extra_context: List[str] = []
    train_df, val_df, test_df, preprocessor = prepare_datasets_simulated(
        df, extra_context, args.seed, encoding_type=args.encoding_type
    )

    csv_dir = os.path.join(args.save_dir, variable)
    model_dir = os.path.join(csv_dir, "residual")
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    if args.mode == "preview":
        preview_cols = [
            c for c in [
                "record_id", "split", "age", "age_group", "gender",
                "condition", "value", "clean_value", "gt_low", "gt_high",
                "normal_low", "normal_high", "is_error", "error_type",
                "donor_context", "corruption_factor",
            ]
            if c in df.columns
        ]
        preview_path = os.path.join(csv_dir, f"preview_simulated_{variable}.csv")
        df[preview_cols].to_csv(preview_path, index=False)
        print(f"Preview saved to: {preview_path}")
        return

    if args.mode == "tune":
        best_path, best_metrics = tune_residual(
            train_df, val_df, preprocessor, args.n_trials,
            args.alpha, args.n_samples, device, model_dir,
        )
        return

    x_train = preprocessor.transform_target(train_df["value"].values)
    c_train = preprocessor.transform_context(train_df)
    x_val = preprocessor.transform_target(val_df["value"].values)
    c_val = preprocessor.transform_context(val_df)

    if args.mode == "train":
        model = _build_model_from_args(preprocessor, c_train.shape[1])
        train_cagpre_residual(
            model, x_train, c_train, x_val, c_val,
            epochs=args.epochs, lr=args.lr, device=device,
            beta=args.beta, batch_size=args.batch_size,
        )
        save_path = os.path.join(model_dir, "best_residual.pt")
        save_model(model, save_path, {
            "hidden_dim": args.hidden_dim,
            "latent_dim": args.latent_dim,
            "num_layers": args.num_layers,
            "lr": args.lr,
            "epochs": args.epochs,
            "beta": args.beta,
            "alpha": args.alpha,
            "n_samples": args.n_samples,
            "batch_size": args.batch_size,
            "model_type": "residual_simulated_aligned",
        })
        metrics = evaluate_residual(
            model, preprocessor, test_df,
            args.n_samples, args.alpha, device,
        )
        result_path = os.path.join(csv_dir, "train_simulated.csv")
        save_results_csv({"CAGPRE_Residual": metrics}, result_path)
        return

    if args.mode == "evaluate":
        load_path = args.load_path or os.path.join(model_dir, "best_residual.pt")
        if not os.path.exists(load_path):
            raise FileNotFoundError(
                f"Model checkpoint not found: {load_path}. "
                "Run --mode train or --mode tune first."
            )

        checkpoint = torch.load(load_path, map_location=device, weights_only=False)
        meta = checkpoint.get("meta", {})
        h_dim = meta.get("hidden_dim", args.hidden_dim)
        l_dim = meta.get("latent_dim", args.latent_dim)
        n_layers = meta.get("num_layers", args.num_layers)
        eval_alpha = meta.get("alpha", args.alpha)
        eval_n_samples = meta.get("n_samples", args.n_samples)

        model = RobustCAGPRECVAE(
            context_dim=c_train.shape[1],
            hidden_dim=h_dim,
            latent_dim=l_dim,
            encoding_type=args.encoding_type,
            cat_dims=preprocessor.get_category_sizes(),
            embed_dim=args.embed_dim,
            num_layers=n_layers,
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)

        metrics = evaluate_residual(
            model, preprocessor, test_df,
            eval_n_samples, eval_alpha, device,
        )
        result_path = os.path.join(csv_dir, "evaluate_simulated.csv")
        save_results_csv({"CAGPRE_Residual": metrics}, result_path)
        return

    if args.mode == "compare":
        csv_path = os.path.join(csv_dir, "compare_simulated.csv")
        results = run_compare_residual_simulated(
            train_df, val_df, test_df, preprocessor,
            args.alpha, args.n_samples, device, args.seed,
            csv_path=csv_path,
            tuned_model_dir=model_dir,
            hyperparams={
                "hidden_dim": args.hidden_dim,
                "latent_dim": args.latent_dim,
                "num_layers": args.num_layers,
                "lr": args.lr,
                "epochs": args.epochs,
                "beta": args.beta,
                "alpha": args.alpha,
                "n_samples": args.n_samples,
                "batch_size": args.batch_size,
            },
            force_train=args.force_train,
        )
        save_results_csv(results, csv_path)
        print(f"\nFinal comparison results saved to: {csv_path}")
        return


if __name__ == "__main__":
    main()
