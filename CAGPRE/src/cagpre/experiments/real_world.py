# -*- coding: utf-8 -*-
"""
CAGPRE Pipeline Residual Version: Version 2
================================
Uses RobustCAGPRECVAE with Huber loss (simpler, single-output decoder)
instead of the dual-headed Gaussian NLL CVAE in cagpre_pipeline_v2.

Shares all infrastructure (data loading, GT, baselines, CLI) with v2.
"""
import argparse
import contextlib
import logging
import math
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
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, RobustScaler
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Import shared infrastructure from v2
# ---------------------------------------------------------------------------
from ..engines.shared import (
    # Reproducibility
    set_seed,
    # Constants
    GENDER_CID, GENDER_CID_REVERSE,
    MEASUREMENT_CONCEPT_IDS, CONDITION_CONCEPT_IDS,
    DEFAULT_CONDITION_LABEL, CONDITION_LABEL_TO_CID,
    GT_FILE_MAP, DB_CONFIG, HEALTHY_BASELINES,
    # Ground Truth
    GroundTruthRow, GroundTruthIndex, load_ground_truth,
    # Data loading
    generate_synthetic_data, get_data_from_omop,
    # Preprocessing & splitting
    TabularPreprocessor, prepare_datasets,
    # Baseline models
    Autoencoder, VanillaVAE, TransformerAutoencoder,
    # Baseline training & evaluation
    train_autoencoder, reconstruction_errors,
    evaluate_isolation_forest, evaluate_lof,
    evaluate_autoencoder_model, evaluate_global_iqr,
    # Metrics
    compute_metrics, compute_iou, compute_all_range_metrics,
    # TopK
    compute_topk_metrics,
    # Model I/O
    save_model, load_model,
    # Utilities
    save_results_csv,
    _load_partial_results, _save_partial_results, _save_range_comparison,
)

try:
    from cagpre_optional_evaluation import PaperEvaluator
except ImportError:
    PaperEvaluator = None

# ===========================================================================
# Residual-Specific Model Architecture
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



class LegacyRobustCAGPRECVAE(nn.Module):
    def __init__(self, context_dim: int, hidden_dim: int, latent_dim: int,
                 encoding_type: str = "onehot", cat_dims: List[int] = None, embed_dim: int = 8):
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
            
        in_dim = actual_context_dim + 1
        self.encoder_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        in_dim = latent_dim + actual_context_dim
        self.decoder_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1)
        )

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
# Residual-Specific Training Logic
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
    batch_size: int = None,
    patience: int = 15,
) -> None:
    """
    PURE UNSUPERVISED TRAINING with Huber Loss (SmoothL1).
    Huber Loss is robust to outliers: treats small errors quadratically
    and large errors (outliers) linearly, preventing the CVAE from
    'collapsing' into the extreme error values found in EHR data.

    Includes early stopping with validation loss monitoring.
    """
    if batch_size is None:
        batch_size = globals().get("args").batch_size if "args" in globals() else 1024

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

    n_train = len(x_train)
    
    global_args = globals().get("args")
    variable = (global_args.variable if global_args and hasattr(global_args, 'variable') else "SBP") or "SBP"
    if variable.upper() in ["GLUCOSE", "PR"]:
        robust_criterion = nn.MSELoss()
    else:
        robust_criterion = nn.SmoothL1Loss(beta=1.0)  # Huber Loss equivalent

    # Pre-load entire dataset to GPU to eliminate PCIe transfer bottleneck
    x_train_t = torch.tensor(x_train, dtype=torch.float32, device=device).unsqueeze(1)
    c_train_t = torch.tensor(c_train, dtype=torch.float32, device=device)
    x_val_t = torch.tensor(x_val, dtype=torch.float32, device=device).unsqueeze(1)
    c_val_t = torch.tensor(c_val, dtype=torch.float32, device=device)

    use_amp = (device != "cpu")
    scaler = torch.amp.GradScaler(enabled=use_amp)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_state = None
    
    # NEW: Track history for KL annealing plotting
    training_history = []
    
    print("\n--- PRE-FLIGHT DEBUG CHECK ---")
    # 1. Check if targets collapsed to zero
    print(f"Target Data (x_train) sample: {x_train_t[:5].flatten().cpu().numpy()}")
    
    # 2. Check for Target Leakage (Does the context matrix look suspiciously like the target?)
    print(f"Context Matrix shape: {c_train_t.shape}")
    
    # 3. Check raw FP32 loss (Bypassing AMP and string formatting)
    model.eval()
    with torch.no_grad():
        test_recon, test_mu, test_logvar = model(x_train_t[:5], c_train_t[:5])
        test_loss = robust_criterion(test_recon, x_train_t[:5])
        print(f"Raw Output sample: {test_recon.flatten().cpu().numpy()}")
        print(f"Raw FP32 Recon Loss: {test_loss.item()}")
    print("------------------------------\n")

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train).to(device)
        epoch_recon, epoch_kl = 0.0, 0.0
        actual_batches = 0
        
        # Calculate cyclical beta for this specific epoch
        current_beta = get_cyclical_beta(epoch, epochs, cycles=4, max_beta=beta, ratio=0.5)

        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            batch_x = x_train_t[idx]
            batch_c = c_train_t[idx]

            with torch.amp.autocast(device_type=device if device != "cpu" else "cpu", enabled=use_amp):
                recon, mu, logvar = model(batch_x, batch_c)

                # Robust Reconstruction Loss
                recon_loss = robust_criterion(recon, batch_x)

                # KL Divergence
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

                loss = recon_loss + (current_beta * kl_loss)

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

        # --- Validation step ---
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for start in range(0, len(x_val), batch_size):
                end = min(start + batch_size, len(x_val))
                x_vb = x_val_t[start:end]
                c_vb = c_val_t[start:end]
                
                with torch.amp.autocast(device_type=device if device != "cpu" else "cpu", enabled=use_amp):
                    recon, mu, logvar = model(x_vb, c_vb)
                    r_loss = robust_criterion(recon, x_vb)
                    k_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                    v_loss = r_loss + beta * k_loss
                if not (torch.isnan(v_loss) or torch.isinf(v_loss)):
                    val_loss += v_loss.item()
                    val_batches += 1

        if val_batches > 0:
            avg_val_loss = val_loss / val_batches
            
            # --- CYCLICAL ANNEALING SAFETY LOCK ---
            # Only trigger early stopping and save the model if we are in the "flat" phase 
            # of the annealing cycle (i.e., when current_beta has reached max beta).
            if current_beta >= beta:
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    epochs_no_improve = 0
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    epochs_no_improve += 1
            else:
                # During warmup phases, we freeze patience and do NOT overwrite the best model
                # because the model is temporarily un-regularized.
                pass
        else:
            avg_val_loss = float("nan")

        if val_batches > 0:
            scheduler.step(avg_val_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            denom = max(actual_batches, 1)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"    Epoch {epoch+1:3d}/{epochs}  "
                  f"recon={epoch_recon/denom:.4f}  kl={epoch_kl/denom:.4f}  "
                  f"val={avg_val_loss:.6f}  lr={current_lr:.1e}  "
                  f"patience={patience - epochs_no_improve}")
                  
        # Save epoch history
        denom = max(actual_batches, 1)
        training_history.append({
            'epoch': epoch,
            'beta': current_beta,
            'recon_loss': epoch_recon / denom,
            'kl_loss': epoch_kl / denom,
            'val_loss': avg_val_loss
        })

        if epochs_no_improve >= patience:
            print(f"    Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
        print(f"    Restored best model (val_loss={best_val_loss:.4f})")
        
    # Save training history to CSV
    try:
        import pandas as pd
        import os
        save_dir = getattr(globals().get("args"), "save_dir", "artifacts_residual")
        os.makedirs(save_dir, exist_ok=True)
        hist_df = pd.DataFrame(training_history)
        hist_path = os.path.join(save_dir, f"training_history_{variable.upper()}.csv")
        hist_df.to_csv(hist_path, index=False)
        print(f"    Saved training history to {hist_path}")
    except Exception as e:
        print(f"    Failed to save training history: {e}")


# ===========================================================================
# Residual-Specific Range Generation (GQE)
# ===========================================================================
def generate_ranges_residual(
    model: RobustCAGPRECVAE,
    context: np.ndarray,
    preprocessor: TabularPreprocessor,
    n_samples: int,
    alpha: float,
    device: str,
    batch_size: int = 4096,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generative Quantile Estimation (GQE) for residual's single-output decoder.
    Uses Dictionary Caching (processing only unique contexts) to reduce
    inference time from hours to milliseconds on massive datasets.
    """
    model.eval()
    
    # 1. Dictionary Caching: Find unique contexts (usually ~160 unique combos)
    unique_contexts, inverse_indices = np.unique(context, axis=0, return_inverse=True)
    n_unique = unique_contexts.shape[0]
    
    unique_low_arr = np.empty(n_unique)
    unique_high_arr = np.empty(n_unique)
    unique_min_arr = np.empty(n_unique)
    unique_max_arr = np.empty(n_unique)
    latent_dim = model.fc_mu.out_features

    # Sub-batch sample generation to avoid CUDA OOM
    sample_chunk_size = 1000

    with torch.no_grad():
        for start in tqdm(range(0, n_unique, batch_size), desc="GQE Generation (Cached)"):
            end = min(start + batch_size, n_unique)
            chunk = unique_contexts[start:end]
            chunk_len = end - start
            c_t = torch.tensor(chunk, dtype=torch.float32, device=device)

            all_decoded = []
            for s_start in range(0, n_samples, sample_chunk_size):
                s_end = min(s_start + sample_chunk_size, n_samples)
                s_count = s_end - s_start

                c_expanded = c_t.unsqueeze(1).expand(-1, s_count, -1).reshape(-1, c_t.shape[1])
                z = torch.randn(chunk_len * s_count, latent_dim, device=device)

                generated = model.decode(z, c_expanded).view(chunk_len, s_count)
                gen_cpu = generated.cpu().numpy()
                gen_real = preprocessor.inverse_target(gen_cpu.flatten()).reshape(chunk_len, s_count)
                all_decoded.append(gen_real)

            decoded = np.concatenate(all_decoded, axis=1)

            # Pure Generative Quantile Estimation (GQE)
            unique_low_arr[start:end] = np.quantile(decoded, alpha / 2.0, axis=1)
            unique_high_arr[start:end] = np.quantile(decoded, 1.0 - alpha / 2.0, axis=1)
            
            # Absolute Survival Limits
            unique_min_arr[start:end] = np.min(decoded, axis=1)
            unique_max_arr[start:end] = np.max(decoded, axis=1)

    # 2. Map the cached results back to the full dataset instantly
    low_arr = unique_low_arr[inverse_indices]
    high_arr = unique_high_arr[inverse_indices]
    min_arr = unique_min_arr[inverse_indices]
    max_arr = unique_max_arr[inverse_indices]

    return low_arr, high_arr, min_arr, max_arr


def generate_ranges_and_scores_residual(
    model: RobustCAGPRECVAE,
    context: np.ndarray,
    preprocessor: TabularPreprocessor,
    observed_values: np.ndarray,
    n_samples: int,
    alpha: float,
    device: str,
    batch_size: int = 4096,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate ranges AND implausibility scores.
    Uses Dictionary Caching and PyTorch Vectorized KDE for massive GPU acceleration.
    """
    model.eval()
    n_rows = context.shape[0]
    
    unique_contexts, inverse_indices = np.unique(context, axis=0, return_inverse=True)
    n_unique = unique_contexts.shape[0]
    
    unique_low = np.empty(n_unique)
    unique_high = np.empty(n_unique)
    unique_min = np.empty(n_unique)
    unique_max = np.empty(n_unique)
    
    # Store all generated samples to run GPU KDE later
    all_unique_samples = np.empty((n_unique, n_samples))

    latent_dim = model.fc_mu.out_features
    sample_chunk_size = 1000

    with torch.no_grad():
        for start in tqdm(range(0, n_unique, batch_size), desc="TopK Generation (Cached)"):
            end = min(start + batch_size, n_unique)
            chunk = unique_contexts[start:end]
            chunk_len = end - start
            c_t = torch.tensor(chunk, dtype=torch.float32, device=device)

            all_decoded = []
            for s_start in range(0, n_samples, sample_chunk_size):
                s_end = min(s_start + sample_chunk_size, n_samples)
                s_count = s_end - s_start
                c_expanded = c_t.unsqueeze(1).expand(-1, s_count, -1).reshape(-1, c_t.shape[1])
                z = torch.randn(chunk_len * s_count, latent_dim, device=device)
                generated = model.decode(z, c_expanded).view(chunk_len, s_count)
                gen_cpu = generated.cpu().numpy()
                gen_real = preprocessor.inverse_target(gen_cpu.flatten()).reshape(chunk_len, s_count)
                all_decoded.append(gen_real)

            decoded = np.concatenate(all_decoded, axis=1)
            all_unique_samples[start:end] = decoded
            
            # GQE Quantiles
            unique_low[start:end] = np.quantile(decoded, alpha / 2.0, axis=1)
            unique_high[start:end] = np.quantile(decoded, 1.0 - alpha / 2.0, axis=1)
            
            # Absolute Survival Limits
            unique_min[start:end] = np.min(decoded, axis=1)
            unique_max[start:end] = np.max(decoded, axis=1)

    # 1. Map bounds back instantly
    low_arr = unique_low[inverse_indices]
    high_arr = unique_high[inverse_indices]
    min_arr = unique_min[inverse_indices]
    max_arr = unique_max[inverse_indices]
    
    # 2. Vectorized GPU KDE Scoring
    samples_t = torch.tensor(all_unique_samples, dtype=torch.float32, device=device)
    obs_t = torch.tensor(observed_values, dtype=torch.float32, device=device)
    inv_idx_t = torch.tensor(inverse_indices, dtype=torch.long, device=device)
    
    scores_t = torch.zeros(n_rows, device=device)
    
    # Scott's Rule for Bandwidth: N^(-1/5) * std
    stds = samples_t.std(dim=1, keepdim=True)
    bws = torch.clamp(stds * (n_samples ** -0.2), min=1e-4) # Prevent division by zero
    
    log_sq2pi = 0.9189385332046727 # log(sqrt(2*pi))
    
    for u_idx in tqdm(range(n_unique), desc="TopK PyTorch KDE Scoring"):
        mask = (inv_idx_t == u_idx)
        if not mask.any(): 
            continue
            
        obs_vals = obs_t[mask] # shape: (K,)
        samps = samples_t[u_idx] # shape: (N_samples,)
        bw = bws[u_idx] # shape: (1,)
        
        # Calculate distances in ultra-safe memory chunks to completely prevent OOM
        chunk_size = 1000  # 1000 * 35,000 samples = ~140 MB VRAM peak
        log_density_list = []
        
        for v_start in range(0, obs_vals.shape[0], chunk_size):
            v_end = min(v_start + chunk_size, obs_vals.shape[0])
            obs_chunk = obs_vals[v_start:v_end]
            
            # Use inplace PyTorch operations to avoid duplicating massive tensors
            diff = samps.unsqueeze(0) - obs_chunk.unsqueeze(1) # (chunk_size, N_samples)
            diff.div_(bw) # inplace division
            
            # Log-PDF of standard normal
            diff.pow_(2) # inplace square
            diff.mul_(-0.5) # inplace multiply
            diff.sub_(log_sq2pi + torch.log(bw)) # inplace subtract
            
            # logsumexp prevents underflow for extreme outliers
            log_density_chunk = torch.logsumexp(diff, dim=1) - math.log(n_samples)
            log_density_list.append(log_density_chunk)
            
            del diff
            del obs_chunk
            
        scores_t[mask] = -torch.cat(log_density_list)
        
    scores = scores_t.cpu().numpy()

    return low_arr, high_arr, min_arr, max_arr, scores


# ===========================================================================
# Residual-Specific Evaluation
# ===========================================================================
def evaluate_residual(
    model: RobustCAGPRECVAE,
    preprocessor: TabularPreprocessor,
    df: pd.DataFrame,
    n_samples: int,
    alpha: float,
    device: str,
    return_ranges: bool = False,
) -> Union[Dict[str, float], Tuple]:
    """Evaluate residual model: generate ranges, compare to GT."""
    c = preprocessor.transform_context(df)
    values = df["value"].values

    pred_low, pred_high, pred_min, pred_max = generate_ranges_residual(
        model, c, preprocessor, n_samples, alpha, device
    )

    # --- Diagnostics ---
    out_df = df.copy()
    out_df["pred_low"] = pred_low
    out_df["pred_high"] = pred_high
    out_df["pred_min"] = pred_min
    out_df["pred_max"] = pred_max

    n_nan_low = np.isnan(pred_low).sum()
    n_nan_high = np.isnan(pred_high).sum()
    n_errors_in_eval = int(df["is_error"].sum())
    n_total = len(df)
    print(f"    [DIAG] Eval set: {n_total} rows, {n_errors_in_eval} errors "
          f"({100*n_errors_in_eval/max(n_total,1):.2f}%)")
    print(f"    [DIAG] Pred range NaN: low={n_nan_low}, high={n_nan_high}")
    if n_nan_low == 0 and n_nan_high == 0:
        print(f"    [DIAG] Pred ranges: low=[{np.min(pred_low):.2f}, {np.max(pred_low):.2f}]  "
              f"high=[{np.min(pred_high):.2f}, {np.max(pred_high):.2f}]")
        print(f"    [DIAG] GT   ranges: low=[{df['gt_low'].min():.2f}, {df['gt_low'].max():.2f}]  "
              f"high=[{df['gt_high'].min():.2f}, {df['gt_high'].max():.2f}]")
        print(f"    [DIAG] Values:      [{df['value'].min():.2f}, {df['value'].max():.2f}]  "
              f"mean={df['value'].mean():.2f}")

    y_pred = (values < pred_low) | (values > pred_high)
    
    # --- Near-Boundary False Positive Analysis ---
    is_error_np = df["is_error"].values
    y_pred_np = y_pred.astype(int)
    fp_mask = (y_pred_np == 1) & (is_error_np == 0)
    n_fp = fp_mask.sum()
    
    if n_fp > 0:
        fp_values = values[fp_mask]
        fp_pred_low = pred_low[fp_mask]
        fp_pred_high = pred_high[fp_mask]
        
        dist_low = np.maximum(0, fp_pred_low - fp_values)
        dist_high = np.maximum(0, fp_values - fp_pred_high)
        dist = np.maximum(dist_low, dist_high)
        
        n_fp_low = (fp_values < fp_pred_low).sum()
        n_fp_high = (fp_values > fp_pred_high).sum()
        
        print(f"    [FP-ANALYSIS] Total False Positives: {n_fp}")
        print(f"    [FP-ANALYSIS] -> FPs below range (low errors): {n_fp_low}")
        print(f"    [FP-ANALYSIS] -> FPs above range (high errors): {n_fp_high}")
        print(f"    [FP-ANALYSIS] -> Median distance to boundary: {np.median(dist):.4f}")
        print(f"    [FP-ANALYSIS] -> Lowest FP value: {fp_values.min():.2f} | Highest FP value: {fp_values.max():.2f}")
        print(f"    [FP-ANALYSIS] -> % of FPs within 5% of boundary: {((dist / np.maximum(1e-5, np.where(dist_low > 0, fp_pred_low, fp_pred_high))) < 0.05).mean() * 100:.1f}%")
    metrics = compute_metrics(df["is_error"].values, y_pred.astype(int))
    metrics.update(compute_all_range_metrics(
        pred_low, pred_high, df["gt_low"].values, df["gt_high"].values, values
    ))

    n_pred_errors = int(y_pred.sum())
    print(f"    [DIAG] Predicted errors: {n_pred_errors} / {n_total}")
    
    # --- Range Diagnostics Tool ---
    diag_df = df.copy()
    diag_df['pred_low'] = pred_low
    diag_df['pred_high'] = pred_high
    diag_df['range_width'] = pred_high - pred_low
    
    group_cols = ["age_group", "gender", "condition"]
    # Ensure columns exist in DataFrame
    group_cols = [col for col in group_cols if col in out_df.columns]
    
    summary_df = out_df.groupby(group_cols, observed=True)[["pred_low", "pred_high", "pred_min", "pred_max"]].mean().reset_index()
    summary_df.to_csv("range_diagnostics.csv", index=False)
    print("    [DIAG] Saved subgroup width analysis to 'range_diagnostics.csv'")

    if return_ranges:
        return metrics, pred_low, pred_high, pred_min, pred_max
    return metrics




def train_conditional_model(
    model: nn.Module,
    c_train: np.ndarray, x_train: np.ndarray,
    c_val: np.ndarray, x_val: np.ndarray,
    epochs: int, lr: float, device: str,
    is_cvae: bool = False,
    batch_size: int = 1024,
) -> None:
    """Train conditional models with Huber loss, AMP gradient scaling,
    and ReduceLROnPlateau for stable convergence on clinical data."""
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6
    )
    
    global_args = globals().get("args")
    variable = (global_args.variable if global_args and hasattr(global_args, 'variable') else "SBP") or "SBP"
    if variable.upper() in ["GLUCOSE", "PR"]:
        criterion = nn.MSELoss()
    else:
        criterion = nn.SmoothL1Loss(beta=1.0)  # Huber
    
    # Pre-load entirely to GPU to eliminate PCIe transfer bottleneck
    x_train_t = torch.tensor(x_train, dtype=torch.float32, device=device).unsqueeze(1)
    c_train_t = torch.tensor(c_train, dtype=torch.float32, device=device)

    use_amp = (device != "cpu")
    scaler = torch.amp.GradScaler(enabled=use_amp)
    n_train = len(x_train)

    best_loss = float("inf")
    best_state = None
    patience_counter = 0
    patience_limit = 20

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train).to(device)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            c_b = c_train_t[idx]
            x_b = x_train_t[idx]

            with torch.amp.autocast(device_type=device if device != "cpu" else "cpu", enabled=use_amp):
                if is_cvae:
                    recon, mu, logvar = model(c_b, x_b)
                    recon_loss = criterion(recon, x_b)
                    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                    # KL annealing: ramp beta linearly over first half of training
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
            print(f"    Early stop at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    print(f"    Final best loss: {best_loss:.4f}")




# ===========================================================================
# Tuning (Improved Scoring + Narrowed Search Space)
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

    best_overall_score = -1.0
    best_path = os.path.join(save_dir, "best_residual.pt")
    best_metrics = {}
    
    # Safely get globals
    global_args = globals().get("args")
    encoding_type = global_args.encoding_type if global_args else "onehot"
    embed_dim = global_args.embed_dim if global_args else 8
    variable = (global_args.variable if global_args and hasattr(global_args, 'variable') else "SBP") or "SBP"

    def objective(trial):
        nonlocal best_overall_score, best_metrics
        
        # 1. Optuna suggests TRAINING parameters (Bayesian optimization)
        hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256])
        latent_dim = trial.suggest_categorical("latent_dim", [8, 16])
        num_layers = trial.suggest_int("num_layers", 2, 4)
        batch_size = trial.suggest_categorical("batch_size", [1024, 4096, 8192])
        lr = trial.suggest_categorical("lr", [1e-3, 5e-4, 2e-4])
        epochs = trial.suggest_categorical("epochs", [60, 80, 100, 120])
        
        # In tune_residual, adjust the beta suggestions based on the variable:
        if variable.upper() in ["GLUCOSE", "PR"]:
            beta = trial.suggest_categorical("beta", [0.01, 0.05, 0.1]) # Lower constraint
        else:
            beta = trial.suggest_categorical("beta", [0.1, 0.15, 0.2, 0.3, 0.5]) # Original

        # Initialize and Train the model ONCE per trial
        model = RobustCAGPRECVAE(
            context_dim=c_train.shape[1],
            hidden_dim=hidden_dim,
            latent_dim=latent_dim, 
            encoding_type=encoding_type, 
            cat_dims=preprocessor.get_category_sizes(), 
            embed_dim=embed_dim, 
            num_layers=num_layers
        )
        
        # Suppress diagnostics only during tuning; the training computation is unchanged.
        with open(os.devnull, "w") as quiet_output:
            with contextlib.redirect_stdout(quiet_output), contextlib.redirect_stderr(quiet_output):
                train_cagpre_residual(
                    model, x_train, c_train, x_val, c_val,
                    epochs, lr, device, beta=beta, batch_size=batch_size
                )

        # 2. Rapidly test multiple INFERENCE parameters (alpha) on the trained model
        # If variable is Glucose or PR, use tighter bounds
        if variable.upper() in ["GLUCOSE", "PR"]:
            alphas_to_test = [0.00001, 0.00002, 0.00005, 0.0001]
        else:
            alphas_to_test = [0.001, 0.003, 0.005, 0.01]
            
        trial_n_samples = trial.suggest_categorical("n_samples", [20000, 35000])
        
        best_trial_score = -1.0
        best_trial_alpha = default_alpha
        
        # SAFETY NET: Pre-fill with empty metrics in case the model is a total failure!
        best_trial_metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "iou": 0.0}

        for test_alpha in alphas_to_test:
            with open(os.devnull, "w") as quiet_output:
                with contextlib.redirect_stdout(quiet_output), contextlib.redirect_stderr(quiet_output):
                    metrics = evaluate_residual(
                        model, preprocessor, val_df, trial_n_samples, test_alpha, device
                    )

            prec_val, rec_val = metrics["precision"], metrics["recall"]
            f1_val, iou_val = metrics["f1"], metrics["iou"]

            if rec_val < 0.05 or iou_val < 0.05:
                current_score = 0.0  # Still kill completely broken models
            else:
                # Standard weighted score
                current_score = (0.3 * prec_val + 0.2 * rec_val + 0.2 * f1_val + 0.3 * iou_val)
                
                # Apply a heavy penalty if it falls into the "Recall = 1.0 / Precision = 0.0" trap
                if rec_val > 0.95 and prec_val < 0.05:
                    current_score *= 0.25

            if current_score >= best_trial_score: # Changed to >= so it at least grabs the last calculated metrics if score stays 0.0
                best_trial_score = current_score
                best_trial_alpha = test_alpha
                best_trial_metrics = metrics

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

        # NEW: Save metrics to trial so they appear in trials_dataframe() CSV
        trial.set_user_attr("precision", best_trial_metrics.get('precision', 0))
        trial.set_user_attr("recall", best_trial_metrics.get('recall', 0))
        trial.set_user_attr("f1", best_trial_metrics.get('f1', 0))
        trial.set_user_attr("iou", best_trial_metrics.get('iou', 0))
        trial.set_user_attr("alpha", best_trial_alpha)

        # 3. Save globally if this is the absolute best combination we've seen
        if best_trial_score > best_overall_score:
            best_overall_score = best_trial_score
            best_metrics = best_trial_metrics
            
            meta = {
                "hidden_dim": hidden_dim,
                "latent_dim": latent_dim,
                "num_layers": num_layers,
                "lr": lr,
                "epochs": epochs,
                "beta": beta,
                "alpha": best_trial_alpha,
                "n_samples": trial_n_samples,
                "model_type": "residual",
            }
            save_model(model, best_path, meta)
            
            print("*** NEW BEST MODEL FOUND ***")

        return best_trial_score

    # Hide only the initial study creation line; keep Optuna's trial results.
    class _SuppressStudyCreation(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return not record.getMessage().startswith("A new study created")

    optuna_logger = optuna.logging.get_logger("optuna")
    creation_filter = _SuppressStudyCreation()
    for handler in optuna_logger.handlers:
        handler.addFilter(creation_filter)
    try:
        study = optuna.create_study(direction="maximize")
    finally:
        for handler in optuna_logger.handlers:
            handler.removeFilter(creation_filter)
    study.optimize(objective, n_trials=n_trials)

    print("\n=== TUNING COMPLETE ===")
    print(f"Best Optuna Trial Score: {study.best_value:.3f}")
    print(f"Best Params: {study.best_params}")
    
    # NEW: Save all Optuna trial results to CSV
    try:
        trials_df = study.trials_dataframe()
        csv_path = os.path.join(save_dir, "optuna_trials_residual.csv")
        trials_df.to_csv(csv_path, index=False)
        print(f"Saved all Optuna trials to {csv_path}")
    except Exception as e:
        print(f"Failed to save Optuna trials: {e}")

    return best_path, best_metrics


# ===========================================================================
# GT-Row-Based Evaluation (for real-world data)
# ===========================================================================
def run_gt_evaluation_residual(
    model: RobustCAGPRECVAE,
    preprocessor: TabularPreprocessor,
    gt_index: GroundTruthIndex,
    n_samples: int,
    alpha: float,
    device: str,
    csv_path: str = "",
    test_df: Optional[pd.DataFrame] = None,
) -> Dict[str, float]:
    """Evaluate model ranges against GT ranges using REAL patient data.

    For each GT group (age_bracket x gender x condition):
      1. Gather all real test patients that fall into that group
      2. Generate per-patient model ranges
      3. Aggregate ranges via mean and median
      4. Compute IoU for both aggregation methods against GT
    """
    print("\n--- GT-Row-Based Evaluation (Residual, Real Data) ---")

    if test_df is None or test_df.empty:
        print("  [WARN] No test data provided, skipping GT evaluation.")
        return {"mean_iou": 0.0, "median_iou": 0.0}

    # Generate model ranges for ALL test patients at once
    c_test = preprocessor.transform_context(test_df)
    pred_low, pred_high, pred_min, pred_max = generate_ranges_residual(
        model, c_test, preprocessor, n_samples, alpha, device
    )

    # Assign each test patient to a GT group
    test_copy = test_df.copy().reset_index(drop=True)
    test_copy["pred_low"] = pred_low
    test_copy["pred_high"] = pred_high

    results_rows = []
    for gt_row in gt_index.rows:
        # Find real patients matching this GT group
        mask = (
            (test_copy["age"] >= gt_row.age_min)
            & (test_copy["age"] < gt_row.age_max)
            & (test_copy["gender"].str.upper() == gt_row.gender.upper())
            & (test_copy["condition"].str.lower() == gt_row.condition.lower())
        )
        group = test_copy[mask]
        n_patients = len(group)

        gt_lo = gt_row.plausible_low
        gt_hi = gt_row.plausible_high

        if n_patients == 0:
            results_rows.append({
                "Gender": gt_row.gender,
                "Age_Range": f"{gt_row.age_min}-{gt_row.age_max - 1}",
                "Condition": gt_row.condition,
                "N_patients": 0,
                "GT_Low": round(gt_lo, 2),
                "GT_High": round(gt_hi, 2),
                "Mean_Low": None, "Mean_High": None, "Mean_IoU": None,
                "Median_Low": None, "Median_High": None, "Median_IoU": None,
            })
            continue

        # --- Mean aggregation ---
        mean_lo = float(group["pred_low"].mean())
        mean_hi = float(group["pred_high"].mean())
        inter_mean = max(0.0, min(mean_hi, gt_hi) - max(mean_lo, gt_lo))
        union_mean = max(mean_hi, gt_hi) - min(mean_lo, gt_lo)
        mean_iou = inter_mean / union_mean if union_mean > 0 else 0.0

        # --- Median aggregation ---
        med_lo = float(group["pred_low"].median())
        med_hi = float(group["pred_high"].median())
        inter_med = max(0.0, min(med_hi, gt_hi) - max(med_lo, gt_lo))
        union_med = max(med_hi, gt_hi) - min(med_lo, gt_lo)
        med_iou = inter_med / union_med if union_med > 0 else 0.0

        results_rows.append({
            "Gender": gt_row.gender,
            "Age_Range": f"{gt_row.age_min}-{gt_row.age_max - 1}",
            "Condition": gt_row.condition,
            "N_patients": n_patients,
            "GT_Low": round(gt_lo, 2),
            "GT_High": round(gt_hi, 2),
            "Mean_Low": round(mean_lo, 2),
            "Mean_High": round(mean_hi, 2),
            "Mean_IoU": round(mean_iou, 4),
            "Median_Low": round(med_lo, 2),
            "Median_High": round(med_hi, 2),
            "Median_IoU": round(med_iou, 4),
        })

    results_df = pd.DataFrame(results_rows)
    # Compute averages only for groups that have patients
    valid = results_df.dropna(subset=["Mean_IoU"])
    avg_mean_iou = float(valid["Mean_IoU"].mean()) if len(valid) > 0 else 0.0
    avg_median_iou = float(valid["Median_IoU"].mean()) if len(valid) > 0 else 0.0
    total_patients = int(results_df["N_patients"].sum())

    print(f"  GT groups: {len(results_df)} | Groups with data: {len(valid)} | "
          f"Total patients: {total_patients:,}")
    print(f"  Average Mean_IoU:   {avg_mean_iou:.4f}")
    print(f"  Average Median_IoU: {avg_median_iou:.4f}")

    if csv_path:
        os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else ".", exist_ok=True)
        summary = pd.concat([
            results_df,
            pd.DataFrame([{
                "Gender": "AVERAGE", "Age_Range": "", "Condition": "",
                "N_patients": total_patients,
                "GT_Low": "", "GT_High": "",
                "Mean_Low": "", "Mean_High": "",
                "Mean_IoU": round(avg_mean_iou, 4),
                "Median_Low": "", "Median_High": "",
                "Median_IoU": round(avg_median_iou, 4),
            }])
        ], ignore_index=True)
        summary.to_csv(csv_path, index=False)
        print(f"  GT evaluation saved to: {csv_path}")

    return {"mean_iou": avg_mean_iou, "median_iou": avg_median_iou}


# ===========================================================================
# Value-Scanning Range Estimation for Non-Generative Baselines
# ===========================================================================
# def value_scan_ranges(
#     model,
#     preprocessor: TabularPreprocessor,
#     train_df: pd.DataFrame,
#     test_df: pd.DataFrame,
#     device: str,
#     model_type: str = "sklearn",
#     n_candidates: int = 5000,
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """Generate plausible ranges by scanning candidate values per context group.
# 
#     For each unique (age_group, gender, condition) context:
#       1. Generate candidate values across the training data range
#       2. Score each (value, context) pair using the model
#       3. Find the range of values the model considers 'plausible'
# 
#     Returns (pred_low, pred_high) arrays for each test row.
#     """
#     # Build context groups
#     test_copy = test_df.copy()
#     test_copy["_age_group"] = pd.cut(
#         test_copy["age"], bins=[0, 19, 29, 39, 49, 59, 69, 79, 89, 200],
#         labels=["0-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80-89", "90+"]
#     )
#     train_copy = train_df.copy()
#     train_copy["_age_group"] = pd.cut(
#         train_copy["age"], bins=[0, 19, 29, 39, 49, 59, 69, 79, 89, 200],
#         labels=["0-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80-89", "90+"]
#     )
# 
#     group_cols = ["_age_group", "gender"] if args.ignore_condition else ["_age_group", "gender", "condition"]
# 
#     # Candidate values span training range with padding
#     train_vals = train_df["value"].values
#     v_min = np.percentile(train_vals, 0.1)
#     v_max = np.percentile(train_vals, 99.9)
#     padding = (v_max - v_min) * 0.2
#     candidate_values = np.linspace(v_min - padding, v_max + padding, n_candidates)
# 
#     # Build group -> range mapping
#     group_ranges = {}
#     unique_groups = test_copy.groupby(group_cols, observed=True).size().reset_index()[group_cols]
# 
#     for _, grp in unique_groups.iterrows():
#         key = tuple(grp[c] for c in group_cols)
# 
#         # Build representative context row
#         matching = train_copy
#         for c, val in zip(group_cols, key):
#             matching = matching[matching[c] == val]
#         if len(matching) == 0:
#             matching = train_copy  # fallback
#         rep_row = matching.iloc[0:1]
# 
#         # Create candidate df: same context, varying values
#         cand_df = pd.DataFrame({
#             "value": candidate_values,
#             "age": [rep_row["age"].values[0]] * n_candidates,
#             "gender": [rep_row["gender"].values[0]] * n_candidates,
#             "condition": [rep_row["condition"].values[0]] * n_candidates,
#         })
#         # Add any other columns needed
#         for col in train_df.columns:
#             if col not in cand_df.columns and col not in ["is_error", "gt_low", "gt_high"]:
#                 cand_df[col] = rep_row[col].values[0]
# 
#         c_cand = preprocessor.transform_context(cand_df)
#         x_cand = preprocessor.transform_target(candidate_values)
#         full_cand = np.hstack([c_cand, x_cand.reshape(-1, 1)])
# 
#         if model_type == "sklearn":
#             scores = model.decision_function(full_cand)
#             # For IF: higher decision_function = more normal
#             # For LOF: higher decision_function = more normal
#             # Inliers have positive scores, outliers negative
#             threshold = np.percentile(scores, 5)  # bottom 5% are outliers
#             plausible_mask = scores >= threshold
#         elif model_type == "ae":
#             # Reconstruction error based
#             model_nn = model
#             model_nn.eval()
#             with torch.no_grad():
#                 x_t = torch.tensor(full_cand, dtype=torch.float32, device=device)
#                 out = model_nn(x_t)
#                 # VanillaVAE returns (recon, mu, logvar); plain AE returns tensor
#                 recon = out[0] if isinstance(out, tuple) else out
#                 errors = ((recon - x_t) ** 2).mean(dim=1).cpu().numpy()
# 
#             # Collect training errors for threshold
#             x_train_full = np.hstack([
#                 preprocessor.transform_context(train_df),
#                 preprocessor.transform_target(train_df["value"].values).reshape(-1, 1)
#             ])
#             with torch.no_grad():
#                 x_tr = torch.tensor(x_train_full[:min(10000, len(x_train_full))],
#                                     dtype=torch.float32, device=device)
#                 out_tr = model_nn(x_tr)
#                 recon_tr = out_tr[0] if isinstance(out_tr, tuple) else out_tr
#                 train_errors = ((recon_tr - x_tr) ** 2).mean(dim=1).cpu().numpy()
#             threshold = np.percentile(train_errors, 95)
#             plausible_mask = errors <= threshold
#         else:
#             raise ValueError(f"Unknown model_type: {model_type}")
# 
#         plausible_vals = candidate_values[plausible_mask]
#         if len(plausible_vals) == 0:
#             # Fallback to training percentiles for this group
#             grp_vals = matching["value"].values if len(matching) > 10 else train_vals
#             group_ranges[key] = (np.percentile(grp_vals, 2.5), np.percentile(grp_vals, 97.5))
#         else:
#             group_ranges[key] = (plausible_vals.min(), plausible_vals.max())
# 
#     # Map ranges back to test rows
#     pred_low = np.empty(len(test_df))
#     pred_high = np.empty(len(test_df))
# 
#     # Global fallback
#     fallback_lo = np.percentile(train_vals, 2.5)
#     fallback_hi = np.percentile(train_vals, 97.5)
# 
#     test_keys = list(zip(
#         test_copy["_age_group"],
#         test_copy["gender"],
#         test_copy["condition"]
#     ))
#     for i, key in enumerate(test_keys):
#         lo, hi = group_ranges.get(key, (fallback_lo, fallback_hi))
#         pred_low[i] = lo
#         pred_high[i] = hi
# 
#     return pred_low, pred_high


def value_scan_ranges(
    model,
    preprocessor: TabularPreprocessor,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    device: str,
    model_type: str = "sklearn",
    n_candidates: int = 5000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate plausible ranges by scanning candidate values per context group.
    Updated to use Binary Search for neural network evaluation.
    """
    # Build context groups
    test_copy = test_df.copy()
    test_copy["_age_group"] = pd.cut(
        test_copy["age"], bins=[0, 19, 29, 39, 49, 59, 69, 79, 89, 200],
        labels=["0-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80-89", "90+"]
    )
    train_copy = train_df.copy()
    train_copy["_age_group"] = pd.cut(
        train_copy["age"], bins=[0, 19, 29, 39, 49, 59, 69, 79, 89, 200],
        labels=["0-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80-89", "90+"]
    )

    group_cols = ["_age_group", "gender"] if args.ignore_condition else ["_age_group", "gender", "condition"]

    # Candidate values span training range with padding
    train_vals = train_df["value"].values
    v_min = np.percentile(train_vals, 0.1)
    v_max = np.percentile(train_vals, 99.9)
    padding = (v_max - v_min) * 0.2
    candidate_values = np.linspace(v_min - padding, v_max + padding, n_candidates)

    # Build group -> range mapping
    group_ranges = {}
    unique_groups = test_copy.groupby(group_cols, observed=True).size().reset_index()[group_cols]

    for _, grp in unique_groups.iterrows():
        key = tuple(grp[c] for c in group_cols)

        # Build representative context row
        matching = train_copy
        for c, val in zip(group_cols, key):
            matching = matching[matching[c] == val]
        if len(matching) == 0:
            matching = train_copy  # fallback
        rep_row = matching.iloc[0:1]

        # Create candidate df: same context, varying values
        cand_df = pd.DataFrame({
            "value": candidate_values,
            "age": [rep_row["age"].values[0]] * n_candidates,
            "gender": [rep_row["gender"].values[0]] * n_candidates,
            "condition": [rep_row["condition"].values[0]] * n_candidates,
        })
        # Add any other columns needed
        for col in train_df.columns:
            if col not in cand_df.columns and col not in ["is_error", "gt_low", "gt_high"]:
                cand_df[col] = rep_row[col].values[0]

        c_cand = preprocessor.transform_context(cand_df)
        x_cand = preprocessor.transform_target(candidate_values)
        full_cand = np.hstack([c_cand, x_cand.reshape(-1, 1)])

        if model_type == "sklearn":
            scores = model.decision_function(full_cand)
            threshold = np.percentile(scores, 5)  # bottom 5% are outliers
            plausible_mask = scores >= threshold
            plausible_vals = candidate_values[plausible_mask]
        elif model_type == "ae":
            model_nn = model
            model_nn.eval()
            
            # 1. Get the threshold from training data
            x_train_full = np.hstack([
                preprocessor.transform_context(train_df),
                preprocessor.transform_target(train_df["value"].values).reshape(-1, 1)
            ])
            with torch.no_grad():
                # Process in chunks to avoid blowing up memory
                train_errors = []
                for chunk_start in range(0, min(10000, len(x_train_full)), 2048):
                    chunk_end = min(chunk_start + 2048, len(x_train_full))
                    x_tr = torch.tensor(x_train_full[chunk_start:chunk_end], dtype=torch.float32, device=device)
                    out_tr = model_nn(x_tr)
                    recon_tr = out_tr[0] if isinstance(out_tr, tuple) else out_tr
                    train_errors.append(((recon_tr - x_tr) ** 2).mean(dim=1).cpu().numpy())
                train_errors = np.concatenate(train_errors)
            threshold = np.percentile(train_errors, 95)

            # 2. Binary Search to find Plausible Bounds
            def get_error(val):
                test_row = np.hstack([c_cand[0:1], preprocessor.transform_target(np.array([val])).reshape(1, 1)])
                with torch.no_grad():
                    x_t = torch.tensor(test_row, dtype=torch.float32, device=device)
                    out = model_nn(x_t)
                    recon = out[0] if isinstance(out, tuple) else out
                    return ((recon - x_t) ** 2).mean().item()

            # Find Lower Bound
            low_bound, high_bound = candidate_values.min(), rep_row["value"].values[0]
            for _ in range(15): # 15 steps is enough for high precision
                mid = (low_bound + high_bound) / 2
                if get_error(mid) <= threshold: high_bound = mid  # Plausible, push left
                else: low_bound = mid  # Implausible, push right
            plausible_min = high_bound

            # Find Upper Bound
            low_bound, high_bound = rep_row["value"].values[0], candidate_values.max()
            for _ in range(15):
                mid = (low_bound + high_bound) / 2
                if get_error(mid) <= threshold: low_bound = mid  # Plausible, push right
                else: high_bound = mid  # Implausible, push left
            plausible_max = low_bound

            plausible_vals = np.array([plausible_min, plausible_max])
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        if len(plausible_vals) == 0:
            # Fallback to training percentiles for this group
            grp_vals = matching["value"].values if len(matching) > 10 else train_vals
            group_ranges[key] = (np.percentile(grp_vals, 2.5), np.percentile(grp_vals, 97.5))
        else:
            group_ranges[key] = (plausible_vals.min(), plausible_vals.max())

    # Map ranges back to test rows
    pred_low = np.empty(len(test_df))
    pred_high = np.empty(len(test_df))

    # Global fallback
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
# Run Compare (all models on simulated data)
# ===========================================================================
def run_compare_residual(
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
    results = _load_partial_results(csv_path) if csv_path else {}

    x_train = preprocessor.transform_target(train_df["value"].values)
    c_train = preprocessor.transform_context(train_df)
    x_val = preprocessor.transform_target(val_df["value"].values)
    c_val = preprocessor.transform_context(val_df)
    x_test = preprocessor.transform_target(test_df["value"].values)
    c_test = preprocessor.transform_context(test_df)

    # Extract hyperparams if provided
    hp = hyperparams or {}
    h_dim = hp.get("hidden_dim", 256)
    l_dim = hp.get("latent_dim", 16)
    n_layers = hp.get("num_layers", args.num_layers if "args" in globals() else 2)
    lr = hp.get("lr", 5e-4)
    epochs = hp.get("epochs", 80)
    beta = hp.get("beta", 0.1)
    eval_alpha = hp.get("alpha", alpha)
    eval_n_samples = hp.get("n_samples", n_samples)
    batch_size = hp.get("batch_size", args.batch_size if "args" in globals() else 1024)

    # --- Residual CAGPRE ---
    residual_low, residual_high, residual_min, residual_max = None, None, None, None
    if "CAGPRE_Residual" not in results:
        t0 = time.time()
        tuned_path = os.path.join(tuned_model_dir, "best_residual.pt") if tuned_model_dir else ""
        if not force_train and tuned_path and os.path.exists(tuned_path):
            print(f"Loading tuned Residual CAGPRE from {tuned_path} ...")
            checkpoint = torch.load(tuned_path, map_location=device, weights_only=False)
            meta = checkpoint.get("meta", {})
            h_dim = meta.get("hidden_dim", h_dim)
            l_dim = meta.get("latent_dim", l_dim)
            n_layers = meta.get("num_layers", n_layers)
            eval_alpha = meta.get("alpha", eval_alpha)
            eval_n_samples = meta.get("n_samples", eval_n_samples)
            print(f"  Tuned hyperparams: hidden={h_dim} latent={l_dim} layers={n_layers} "
                  f"alpha={eval_alpha} n_samples={eval_n_samples}")

            # Auto-detect saved model's context_dim from encoder weights
            saved_input_dim = checkpoint["state_dict"]["encoder_net.0.weight"].shape[1]
            saved_context_dim = saved_input_dim - 1  # encoder input = 1 (value) + context_dim

            if saved_context_dim != c_train.shape[1]:
                print(f"  [WARN] Saved model context_dim={saved_context_dim} != "
                      f"current preprocessor context_dim={c_train.shape[1]}")
                print(f"  Re-fitting preprocessor to match saved model ...")
                cat_cols_compat = ["gender"] if args.ignore_condition else ["gender", "condition"]
                num_cols_compat = ["age"]
                preprocessor = TabularPreprocessor(cat_cols=cat_cols_compat, num_cols=num_cols_compat, encoding_type=args.encoding_type)
                preprocessor.fit(train_df)
                c_train = preprocessor.transform_context(train_df)
                c_val = preprocessor.transform_context(val_df)
                c_test = preprocessor.transform_context(test_df)
                print(f"  New context_dim={c_train.shape[1]} (matches saved {saved_context_dim})")

            is_legacy = "num_layers" not in checkpoint.get("meta", {}) and "num_layers" not in checkpoint
            if is_legacy:
                cagpre = LegacyRobustCAGPRECVAE(context_dim=c_train.shape[1], hidden_dim=h_dim, latent_dim=l_dim, encoding_type=args.encoding_type, cat_dims=preprocessor.get_category_sizes(), embed_dim=args.embed_dim)
            else:
                cagpre = RobustCAGPRECVAE(context_dim=c_train.shape[1], hidden_dim=h_dim, latent_dim=l_dim, encoding_type=args.encoding_type, cat_dims=preprocessor.get_category_sizes(), embed_dim=args.embed_dim, num_layers=n_layers)
            cagpre.load_state_dict(checkpoint["state_dict"])
            cagpre.to(device)
            t1 = time.time()
            results["CAGPRE_Residual"], residual_low, residual_high, residual_min, residual_max = evaluate_residual(
                cagpre, preprocessor, test_df, eval_n_samples, eval_alpha, device,
                return_ranges=True,
            )
            train_time = t1 - t0
            test_time = time.time() - t1
        else:
            print(f"Training Residual CAGPRE (Huber CVAE) from scratch with params: "
                  f"hidden={h_dim}, latent={l_dim}, layers={n_layers}, lr={lr}, epochs={epochs}, beta={beta}, batch_size={batch_size} ...")
            cagpre = RobustCAGPRECVAE(
                context_dim=c_train.shape[1],
                hidden_dim=h_dim,
                latent_dim=l_dim,
                encoding_type=args.encoding_type if "args" in globals() else "onehot",
                cat_dims=preprocessor.get_category_sizes(),
                embed_dim=args.embed_dim if "args" in globals() else 8,
                num_layers=n_layers,
            )
            train_cagpre_residual(
                cagpre, x_train, c_train, x_val, c_val,
                epochs=epochs, lr=lr, device=device, beta=beta, batch_size=batch_size
            )
            t1 = time.time()
            results["CAGPRE_Residual"], residual_low, residual_high, residual_min, residual_max = evaluate_residual(
                cagpre, preprocessor, test_df, eval_n_samples, eval_alpha, device,
                return_ranges=True,
            )
            train_time = t1 - t0
            test_time = time.time() - t1
        results["CAGPRE_Residual"]["train_time"] = round(train_time, 2)
        results["CAGPRE_Residual"]["test_time"] = round(test_time, 2)
        results["CAGPRE_Residual"]["time_sec"] = round(time.time() - t0, 2)
        print(f"  CAGPRE_Residual done ({results['CAGPRE_Residual']['time_sec']}s): "
              f"{results['CAGPRE_Residual']}")
        if csv_path:
            _save_partial_results(results, csv_path)
    else:
        print("  [RESUME] Skipping CAGPRE_Residual (already done)")

    # --- Global IQR ---
    iqr_low_arr, iqr_high_arr = None, None
    if "Global_IQR" not in results:
        print("Evaluating Global IQR ...")
        t0 = time.time()
        results["Global_IQR"], iqr_low_arr, iqr_high_arr = evaluate_global_iqr(
            train_df, test_df, return_ranges=True
        )
        t1 = time.time()
        results["Global_IQR"]["train_time"] = 0.0
        results["Global_IQR"]["test_time"] = round(t1 - t0, 2)
        results["Global_IQR"]["time_sec"] = round(t1 - t0, 2)
        print(f"  Global IQR done ({results['Global_IQR']['time_sec']}s): {results['Global_IQR']}")
        if csv_path:
            _save_partial_results(results, csv_path)
    else:
        print("  [RESUME] Skipping Global_IQR")

    full_train = np.hstack([c_train, x_train.reshape(-1, 1)])
    full_val = np.hstack([c_val, x_val.reshape(-1, 1)])
    full_test = np.hstack([c_test, x_test.reshape(-1, 1)])

    # Subsample for sklearn if very large (reduced for faster baseline training)
    MAX_SKLEARN_ROWS = 50_000
    if full_train.shape[0] > MAX_SKLEARN_ROWS:
        print(f"  [NOTE] Subsampling for sklearn: {full_train.shape[0]:,} -> {MAX_SKLEARN_ROWS:,}")
        rng = np.random.default_rng(seed)
        sk_idx = rng.choice(full_train.shape[0], MAX_SKLEARN_ROWS, replace=False)
        sk_train = full_train[sk_idx]
    else:
        sk_train = full_train

    # --- Collect all baseline ranges for the range comparison CSV ---
    model_ranges = {}

    # --- Isolation Forest (value-scanning range estimation) ---
    if "Isolation_Forest" not in results:
        from sklearn.ensemble import IsolationForest as IF
        print("Training Isolation Forest (n_estimators=100, value-scanning) ...")
        t0 = time.time()
        iso = IF(n_estimators=100, contamination=0.05, random_state=seed,
                 max_features=1.0, bootstrap=True)
        iso.fit(sk_train)
        t1 = time.time()
        # Value-scanning for range estimation
        if_low, if_high = value_scan_ranges(
            iso, preprocessor, train_df, test_df, device, model_type="sklearn"
        )
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
        print(f"  Isolation Forest done ({results['Isolation_Forest']['time_sec']}s): "
              f"{results['Isolation_Forest']}")
        model_ranges["Isolation_Forest"] = (if_low, if_high)
        if csv_path:
            _save_partial_results(results, csv_path)
    else:
        print("  [RESUME] Skipping Isolation_Forest")

    # --- LOF (value-scanning range estimation) ---
    if "LOF" not in results:
        from sklearn.neighbors import LocalOutlierFactor as LOFModel
        print("Training LOF (n_neighbors=20, value-scanning) ...")
        t0 = time.time()
        lof = LOFModel(n_neighbors=20, novelty=True, contamination=0.05,
                       metric="manhattan")
        lof.fit(sk_train)
        t1 = time.time()
        lof_low, lof_high = value_scan_ranges(
            lof, preprocessor, train_df, test_df, device, model_type="sklearn"
        )
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
    else:
        print("  [RESUME] Skipping LOF")

    # --- Standard Autoencoder (value-scanning range estimation) ---
    if "Autoencoder" not in results:
        print("Training Standard Autoencoder (value-scanning) ...")
        t0 = time.time()
        ae = Autoencoder(input_dim=full_train.shape[1], hidden_dim=128)
        train_autoencoder(ae, full_train, full_val, 50, 5e-4, device, batch_size=4096)
        t1 = time.time()
        ae_low, ae_high = value_scan_ranges(
            ae, preprocessor, train_df, test_df, device, model_type="ae"
        )
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
        print(f"  Autoencoder done ({results['Autoencoder']['time_sec']}s): "
              f"{results['Autoencoder']}")
        model_ranges["Autoencoder"] = (ae_low, ae_high)
        if csv_path:
            _save_partial_results(results, csv_path)
    else:
        print("  [RESUME] Skipping Autoencoder")

    # --- Vanilla VAE (value-scanning range estimation) ---
    if "VAE" not in results:
        print("Training Vanilla VAE (value-scanning) ...")
        t0 = time.time()
        vae = VanillaVAE(input_dim=full_train.shape[1], hidden_dim=128, latent_dim=16)
        train_autoencoder(vae, full_train, full_val, 50, 5e-4, device, is_vae=True, batch_size=4096)
        t1 = time.time()
        vae_low, vae_high = value_scan_ranges(
            vae, preprocessor, train_df, test_df, device, model_type="ae"
        )
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
    else:
        print("  [RESUME] Skipping VAE")

    # --- Transformer AE (value-scanning range estimation) ---
    if "Transformer_AE" not in results:
        print("Training Transformer Autoencoder (value-scanning) ...")
        t0 = time.time()
        tae = TransformerAutoencoder(input_dim=full_train.shape[1], d_model=32, n_heads=4, n_layers=2)
        train_autoencoder(tae, full_train, full_val, 50, 5e-4, device, batch_size=4096)
        t1 = time.time()
        tae_low, tae_high = value_scan_ranges(
            tae, preprocessor, train_df, test_df, device, model_type="ae"
        )
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
        print(f"  Transformer AE done ({results['Transformer_AE']['time_sec']}s): "
              f"{results['Transformer_AE']}")
        model_ranges["Transformer_AE"] = (tae_low, tae_high)
        if csv_path:
            _save_partial_results(results, csv_path)
    else:
        print("  [RESUME] Skipping Transformer_AE")

    # --- Save range comparison CSV with ALL model ranges ---
    if csv_path and (residual_low is not None or iqr_low_arr is not None or model_ranges):
        range_csv = csv_path.replace("compare_", "range_comparison_")
        _save_range_comparison(
            test_df, residual_low, residual_high, None,
            iqr_low_arr, iqr_high_arr, range_csv,
            model_ranges=model_ranges,
        )

    return results


# ===========================================================================
# Main
# ===========================================================================
# Global configuration so functions can resolve dynamic arguments
args = None

def main() -> None:
    global args
    parser = argparse.ArgumentParser(
        description="CAGPRE Pipeline (Residual Variant) - Huber-loss CVAE"
    )
    parser.add_argument("--mode", choices=["fetch_data", "tune", "train", "evaluate", "compare", "preview", "scalability", "topk", "generate_ranges"],
                        required=True)
    parser.add_argument("--dataset", choices=["simulated", "real"], required=True)
    parser.add_argument("--variable", default=None, help="SBP, DBP, PR, CH, Glucose, NT-proBNP, Bilirubin_total, C-reactive_protein, Cholesterol_HDL, Cholesterol_LDL, Fibrinogen, Uric_acid, Alanine_aminotransferase, Albumin, Aspartate_aminotransferase, ESR")
    parser.add_argument("--gt-dir", default=".", help="Directory containing GT CSV files")
    parser.add_argument("--save-dir", default="artifacts_residual")
    parser.add_argument("--load-path", default=None)
    parser.add_argument("--row-limit", type=int, default=None)
    # Database connection
    parser.add_argument("--db-host", default=None)
    parser.add_argument("--db-port", type=int, default=5432)
    parser.add_argument("--db-name", default=None)
    parser.add_argument("--db-user", default=None)
    parser.add_argument("--db-password", default=None)
    parser.add_argument("--schema", default="public")
    # Model params
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.005, help="Alpha percentile for Pure GQE")
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-synthetic", type=int, default=20000)
    parser.add_argument("--error-rate", type=float, default=0.05)
    # Residual-specific
    parser.add_argument("--beta", type=float, default=0.1, help="KL weight for Residual CVAE")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--num-layers", type=int, default=2, help="Number of hidden layers in encoder and decoder")
    # Encoding arguments
    parser.add_argument("--encoding-type", choices=["onehot", "embedding"], default="onehot", help="How to encode categoricals")
    parser.add_argument("--embed-dim", type=int, default=8, help="Embedding dimension")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size for training and inference")
    # Condition filter
    parser.add_argument("--condition-filter", default=None,
                        help="Filter data to a specific condition, e.g. 'Type 1 diabetes mellitus'")
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if not args.variable:
        raise ValueError("Provide --variable (SBP, DBP, PR, CH, Glucose)")
    variable = args.variable
    args.ignore_condition = variable.upper() not in ["SBP", "DBP", "PR", "CH", "GLUCOSE"]

    # --- Load Ground Truth ---
    gt_index = load_ground_truth(variable, args.gt_dir)

    # --- Load Data ---
    if args.dataset == "real":
        db_params = {
            "host": args.db_host or DB_CONFIG["host"],
            "port": args.db_port or DB_CONFIG["port"],
            "dbname": args.db_name or DB_CONFIG["dbname"],
            "user": args.db_user or DB_CONFIG["user"],
            "password": args.db_password or DB_CONFIG["password"],
        }
        schema = args.schema if args.schema != "public" else DB_CONFIG.get("schema", "public")
        df = get_data_from_omop(
            variable=variable, db_params=db_params, schema=schema,
            gt_index=gt_index, row_limit=args.row_limit,
        )
    else:
        df = generate_synthetic_data(gt_index, args.n_synthetic, args.error_rate, variable, args.seed)

    extra_context: List[str] = []
    print(f"Dataset: {args.dataset} | Variable: {variable} | "
          f"Records: {len(df)} | Errors: {df['is_error'].sum()}")

    # --- fetch_data mode ---
    if args.mode == "fetch_data":
        if args.dataset != "real":
            print("  [INFO] fetch_data mode is intended for --dataset real. For simulated data, generation is instantaneous.")
        print(f"\n  Data successfully fetched and cached for {variable}.")
        return

    # --- Preview mode ---
    if args.mode == "preview":
        csv_dir = os.path.join(args.save_dir, variable.upper())
        os.makedirs(csv_dir, exist_ok=True)
        preview_path = os.path.join(csv_dir, f"preview_{args.dataset}_{variable.upper()}.csv")
        normal = df[df["is_error"] == 0].head(150)
        errors = df[df["is_error"] == 1].head(50)
        sample = pd.concat([normal, errors]).sample(frac=1, random_state=args.seed).reset_index(drop=True)
        sample.to_csv(preview_path, index=False)
        print(f"\nPreview saved to: {preview_path}")
        return

    # --- Scalability mode ---
    if args.mode == "scalability":
        percentages = [1, 5, 10, 20, 50, 100]
        tuned_dir = os.path.join(args.save_dir, variable.upper(), "residual")
        best_model_path = args.load_path if args.load_path else os.path.join(tuned_dir, "best_residual.pt")

        hyperparams = {
            "hidden_dim": args.hidden_dim,
            "latent_dim": args.latent_dim,
            "num_layers": args.num_layers,
            "lr": args.lr,
            "epochs": args.epochs,
            "beta": args.beta,
            "alpha": args.alpha,
            "n_samples": args.n_samples,
        }

        if os.path.exists(best_model_path):
            try:
                checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
                meta = checkpoint.get("meta", {})
                if meta:
                    print(f"\n  [SCALABILITY] Found best model checkpoint at {best_model_path}.")
                    print(f"  Loading hyperparams: {meta}")
                    hyperparams["hidden_dim"] = meta.get("hidden_dim", hyperparams["hidden_dim"])
                    hyperparams["latent_dim"] = meta.get("latent_dim", hyperparams["latent_dim"])
                    hyperparams["num_layers"] = meta.get("num_layers", hyperparams["num_layers"])
                    hyperparams["lr"] = meta.get("lr", hyperparams["lr"])
                    hyperparams["epochs"] = meta.get("epochs", hyperparams["epochs"])
                    hyperparams["beta"] = meta.get("beta", hyperparams["beta"])
                    hyperparams["alpha"] = meta.get("alpha", hyperparams["alpha"])
                    hyperparams["n_samples"] = meta.get("n_samples", hyperparams["n_samples"])
            except Exception as e:
                print(f"  [WARN] Failed to load best model metadata: {e}. Using CLI defaults.")
        else:
            print(f"\n  [SCALABILITY] No tuned model found at {best_model_path}. Using CLI default hyperparams.")

        N_total = len(df)
        print(f"  [SCALABILITY] Full dataset size: {N_total:,} records")
        all_rows = []

        for pct in percentages:
            n = max(10, int(round(N_total * (pct / 100.0))))
            print(f"\n{'='*70}\nSCALABILITY: {variable.upper()} | {pct}% ({n:,} / {N_total:,} records)\n{'='*70}")
            set_seed(args.seed)
            if pct == 100 or n >= N_total:
                df_n = df.copy()
            else:
                df_n = df.sample(n=n, random_state=args.seed).reset_index(drop=True)

            train_n, val_n, test_n, prep_n = prepare_datasets(
                df_n, extra_context, args.seed, encoding_type=args.encoding_type, ignore_condition=args.ignore_condition
            )
            results_n = run_compare_residual(
                train_n, val_n, test_n, prep_n,
                alpha=hyperparams["alpha"],
                n_samples=hyperparams["n_samples"],
                device=device,
                seed=args.seed,
                hyperparams=hyperparams,
                force_train=True,
            )
            for model_name, metrics in results_n.items():
                iou_val = metrics.get("iou")
                all_rows.append({
                    "pct_dataset": pct,
                    "n_records": n,
                    "model": model_name,
                    "train_time": round(metrics.get("train_time", 0.0), 2),
                    "test_time": round(metrics.get("test_time", 0.0), 2),
                    "time_sec": round(metrics.get("time_sec", 0.0), 2),
                    "precision": round(metrics.get("precision", 0.0), 4),
                    "recall": round(metrics.get("recall", 0.0), 4),
                    "f1": round(metrics.get("f1", 0.0), 4),
                    "iou": round(iou_val, 4) if isinstance(iou_val, (int, float)) else "",
                })
        csv_dir = os.path.join(args.save_dir, variable.upper())
        os.makedirs(csv_dir, exist_ok=True)
        csv_path = os.path.join(csv_dir, f"scalability_{args.dataset}.csv")
        pd.DataFrame(all_rows).to_csv(csv_path, index=False)
        print(f"\nScalability saved to: {csv_path}")
        return

    # --- Standard pipeline ---
    # Fit preprocessor on FULL data (all conditions) so context_dim matches saved models
    train_df, val_df, test_df, preprocessor = prepare_datasets(df, extra_context, args.seed, encoding_type=args.encoding_type, ignore_condition=args.ignore_condition)

    # --- Apply condition filter AFTER preprocessor is fitted ---
    if args.condition_filter:
        cond = args.condition_filter
        available_conditions = list(df["condition"].unique())
        before_train, before_val, before_test = len(train_df), len(val_df), len(test_df)
        train_df = train_df[train_df["condition"] == cond].copy().reset_index(drop=True)
        val_df = val_df[val_df["condition"] == cond].copy().reset_index(drop=True)
        test_df = test_df[test_df["condition"] == cond].copy().reset_index(drop=True)
        total_after = len(train_df) + len(val_df) + len(test_df)
        if total_after == 0:
            raise ValueError(
                f"No records found for condition='{cond}'. "
                f"Available conditions: {available_conditions}"
            )
        print(f"\n  Condition filter: '{cond}'")
        print(f"    train: {before_train:,} -> {len(train_df):,}")
        print(f"    val:   {before_val:,} -> {len(val_df):,}")
        print(f"    test:  {before_test:,} -> {len(test_df):,}")

    save_root = os.path.join(args.save_dir, variable.upper(), "residual")
    csv_dir = os.path.join(args.save_dir, variable.upper())
    os.makedirs(csv_dir, exist_ok=True)

    if args.mode == "compare":
        csv_path = os.path.join(csv_dir, f"compare_{args.dataset}.csv")
        tuned_dir = os.path.join(args.save_dir, variable.upper(), "residual")

        if args.dataset == "real":
            # GT-row evaluation
            x_train = preprocessor.transform_target(train_df["value"].values)
            c_train = preprocessor.transform_context(train_df)
            x_val = preprocessor.transform_target(val_df["value"].values)
            c_val = preprocessor.transform_context(val_df)

            tuned_path = os.path.join(tuned_dir, "best_residual.pt")
            if os.path.exists(tuned_path):
                print(f"Loading tuned Residual model from {tuned_path} ...")
                checkpoint = torch.load(tuned_path, map_location=device, weights_only=False)
                
                # Support both old and new checkpoint formats
                if "meta" in checkpoint:
                    meta = checkpoint["meta"]
                    h_dim = meta.get("hidden_dim", args.hidden_dim)
                    l_dim = meta.get("latent_dim", args.latent_dim)
                    n_layers = meta.get("num_layers", args.num_layers)
                    best_alpha = meta.get("alpha", args.alpha)
                    best_n_samples = meta.get("n_samples", args.n_samples)
                else:
                    h_dim = checkpoint.get("hidden_dim", args.hidden_dim)
                    l_dim = checkpoint.get("latent_dim", args.latent_dim)
                    n_layers = checkpoint.get("num_layers", args.num_layers)
                    best_alpha = checkpoint.get("alpha", args.alpha)
                    best_n_samples = checkpoint.get("n_samples", args.n_samples)

                # Auto-detect saved model's context_dim from encoder weights
                saved_input_dim = checkpoint["state_dict"]["encoder_net.0.weight"].shape[1]
                saved_context_dim = saved_input_dim - 1  # encoder input = 1 (value) + context_dim

                if saved_context_dim != c_train.shape[1]:
                    print(f"  [WARN] Saved model context_dim={saved_context_dim} != "
                          f"current preprocessor context_dim={c_train.shape[1]}")
                    print(f"  Re-fitting preprocessor to match saved model ...")
                    # Re-fit preprocessor without is_medicated to match saved model
                    cat_cols_compat = ["gender"] if args.ignore_condition else ["gender", "condition"]
                    num_cols_compat = ["age"] + extra_context
                    preprocessor = TabularPreprocessor(cat_cols=cat_cols_compat, num_cols=num_cols_compat, encoding_type=args.encoding_type)
                    preprocessor.fit(train_df)
                    c_train = preprocessor.transform_context(train_df)
                    c_val = preprocessor.transform_context(val_df)
                    print(f"  New context_dim={c_train.shape[1]} (should match {saved_context_dim})")

                is_legacy = "num_layers" not in checkpoint.get("meta", {}) and "num_layers" not in checkpoint
                if is_legacy:
                    cagpre_model = LegacyRobustCAGPRECVAE(context_dim=c_train.shape[1], hidden_dim=h_dim, latent_dim=l_dim, encoding_type=args.encoding_type, cat_dims=preprocessor.get_category_sizes(), embed_dim=args.embed_dim)
                else:
                    cagpre_model = RobustCAGPRECVAE(context_dim=c_train.shape[1], hidden_dim=h_dim, latent_dim=l_dim, encoding_type=args.encoding_type, cat_dims=preprocessor.get_category_sizes(), embed_dim=args.embed_dim, num_layers=n_layers)
                cagpre_model.load_state_dict(checkpoint["state_dict"])
                cagpre_model.to(device)
            else:
                print("Training Residual CAGPRE from scratch for real data ...")
                best_alpha = args.alpha
                best_n_samples = args.n_samples
                cagpre_model = RobustCAGPRECVAE(
                    context_dim=c_train.shape[1],
                    hidden_dim=args.hidden_dim,
                    latent_dim=args.latent_dim, encoding_type=args.encoding_type, cat_dims=preprocessor.get_category_sizes(), embed_dim=args.embed_dim, num_layers=args.num_layers)
                train_cagpre_residual(
                    cagpre_model, x_train, c_train, x_val, c_val,
                    epochs=args.epochs, lr=args.lr, device=device, beta=args.beta,
                )

            gt_csv = os.path.join(csv_dir, f"gt_eval_{args.dataset}.csv")
            gt_results = run_gt_evaluation_residual(
                cagpre_model, preprocessor, gt_index,
                best_n_samples, best_alpha, device, csv_path=gt_csv,
                test_df=test_df,
            )
            print(f"\n  GT Evaluation Mean_IoU: {gt_results['mean_iou']:.4f}  "
                  f"Median_IoU: {gt_results['median_iou']:.4f}")

        results = run_compare_residual(
            train_df, val_df, test_df, preprocessor, args.alpha, args.n_samples,
            device, args.seed, csv_path=csv_path, tuned_model_dir=tuned_dir,
        )
        save_results_csv(results, csv_path)
        return

    # --- TopK mode ---
    if args.mode == "topk":
        x_train = preprocessor.transform_target(train_df["value"].values)
        c_train = preprocessor.transform_context(train_df)
        x_val = preprocessor.transform_target(val_df["value"].values)
        c_val = preprocessor.transform_context(val_df)

        tuned_dir = os.path.join(args.save_dir, variable.upper(), "residual")
        tuned_path = os.path.join(tuned_dir, "best_residual.pt")
        if os.path.exists(tuned_path):
            print(f"Loading tuned Residual model from {tuned_path} ...")
            checkpoint = torch.load(tuned_path, map_location=device, weights_only=False)
            
            # Support both old and new checkpoint formats
            if "meta" in checkpoint:
                meta = checkpoint["meta"]
                h_dim = meta.get("hidden_dim", args.hidden_dim)
                l_dim = meta.get("latent_dim", args.latent_dim)
                n_layers = meta.get("num_layers", args.num_layers)
                best_alpha = meta.get("alpha", args.alpha)
                best_n_samples = meta.get("n_samples", args.n_samples)
            else:
                h_dim = checkpoint.get("hidden_dim", args.hidden_dim)
                l_dim = checkpoint.get("latent_dim", args.latent_dim)
                n_layers = checkpoint.get("num_layers", args.num_layers)
                best_alpha = checkpoint.get("alpha", args.alpha)
                best_n_samples = checkpoint.get("n_samples", args.n_samples)
            print(f"  Tuned hyperparams: hidden={h_dim} latent={l_dim} "
                  f"alpha={best_alpha} n_samples={best_n_samples}")

            # Auto-detect saved model's context_dim
            saved_input_dim = checkpoint["state_dict"]["encoder_net.0.weight"].shape[1]
            saved_context_dim = saved_input_dim - 1
            if saved_context_dim != c_train.shape[1]:
                print(f"  [WARN] Saved model context_dim={saved_context_dim} != current {c_train.shape[1]}")
                print(f"  Re-fitting preprocessor to match saved model ...")
                cat_cols_compat = ["gender"] if args.ignore_condition else ["gender", "condition"]
                num_cols_compat = ["age"] + extra_context
                preprocessor = TabularPreprocessor(cat_cols=cat_cols_compat, num_cols=num_cols_compat, encoding_type=args.encoding_type)
                preprocessor.fit(train_df)
                c_train = preprocessor.transform_context(train_df)
                c_val = preprocessor.transform_context(val_df)
        else:
            print("No tuned model found. Training Residual CAGPRE from scratch ...")
            best_alpha = args.alpha
            best_n_samples = args.n_samples
            cagpre_model = RobustCAGPRECVAE(
                context_dim=c_train.shape[1],
                hidden_dim=args.hidden_dim,
                latent_dim=args.latent_dim, encoding_type=args.encoding_type, cat_dims=preprocessor.get_category_sizes(), embed_dim=args.embed_dim, num_layers=args.num_layers)
            train_cagpre_residual(
                cagpre_model, x_train, c_train, x_val, c_val,
                epochs=args.epochs, lr=args.lr, device=device, beta=args.beta,
            )

        # Compute scores on test set
        c_test = preprocessor.transform_context(test_df)
        values = test_df["value"].values
        print(f"\n  Computing implausibility scores for {len(test_df)} test rows...")
        pred_low, pred_high, pred_min, pred_max, imp_scores = generate_ranges_and_scores_residual(
            cagpre_model, c_test, preprocessor, values,
            best_n_samples, best_alpha, device
        )
        print(f"  Total errors in test: {int(test_df['is_error'].sum())} / {len(test_df)} "
              f"({100 * test_df['is_error'].mean():.2f}%)")
        print(f"  Score range: [{np.min(imp_scores):.2f}, {np.max(imp_scores):.2f}], "
              f"median={np.median(imp_scores):.2f}")

        # Save raw per-patient implausibility scores CSV
        topk_save_dir = os.path.join(csv_dir, "topk")
        os.makedirs(topk_save_dir, exist_ok=True)
        _ctx_cols = [c for c in ["value", "age", "age_group", "gender", "condition",
                                  "is_error", "error_type", "gt_low", "gt_high"] if c in test_df.columns]
        raw_df = test_df[_ctx_cols].copy().reset_index(drop=True)
        raw_df["pred_low"]             = pred_low
        raw_df["pred_high"]            = pred_high
        raw_df["implausibility_score"] = imp_scores
        raw_df["is_flagged_by_range"]  = (
            (test_df["value"].values < pred_low) |
            (test_df["value"].values > pred_high)
        ).astype(int)
        raw_df_sorted = raw_df.sort_values("implausibility_score",
                                            ascending=False).reset_index(drop=True)
        raw_df_sorted.insert(0, "rank", raw_df_sorted.index + 1)
        raw_scores_path = os.path.join(topk_save_dir, f"raw_scores_{variable}.csv")
        raw_df_sorted.to_csv(raw_scores_path, index=False)
        print(f"  Raw per-patient scores saved → {raw_scores_path}")

        # --- Run PaperEvaluator ---
        print("\n" + "="*50)
        print("Running comprehensive PaperEvaluator report...")
        eval_df = raw_df.copy()
        eval_df.rename(columns={
            'pred_low': 'pred_min', 
            'pred_high': 'pred_max', 
            'is_error': 'is_anomaly'
        }, inplace=True)
        if PaperEvaluator is None:
            raise RuntimeError("Optional detailed report requires the unpublished cagpre_optional_evaluation.py module; this file was not provided.")
        evaluator = PaperEvaluator(eval_df)
        evaluator.run_all_experiments()
        print("="*50 + "\n")

        # Compute @topK metrics
        topk_df = compute_topk_metrics(
            y_true=test_df["is_error"].values,
            scores=imp_scores,
            gt_low=test_df["gt_low"].values,
            gt_high=test_df["gt_high"].values,
            pred_low=pred_low,
            pred_high=pred_high,
            save_dir=topk_save_dir,
            variable=variable,
        )
        print(f"\nTopK results saved to: {topk_save_dir}")
        return

    # --- tune / train / evaluate modes ---
    x_train = preprocessor.transform_target(train_df["value"].values)
    c_train = preprocessor.transform_context(train_df)
    x_val = preprocessor.transform_target(val_df["value"].values)
    c_val = preprocessor.transform_context(val_df)

    if args.mode == "tune":
        best_path, metrics = tune_residual(
            train_df, val_df, preprocessor, args.n_trials, args.alpha,
            args.n_samples, device, save_root
        )
        results = {"CAGPRE_Residual_best": metrics}
        csv_path = os.path.join(csv_dir, f"tune_{args.dataset}.csv")
        save_results_csv(results, csv_path)
        print(f"Best model saved to: {best_path}")
        return

    if args.mode == "train":
        model = RobustCAGPRECVAE(
            context_dim=c_train.shape[1],
            hidden_dim=args.hidden_dim,
            latent_dim=args.latent_dim, encoding_type=args.encoding_type, cat_dims=preprocessor.get_category_sizes(), embed_dim=args.embed_dim, num_layers=args.num_layers)
        train_cagpre_residual(
            model, x_train, c_train, x_val, c_val,
            epochs=args.epochs, lr=args.lr, device=device, beta=args.beta,
        )
        save_path = os.path.join(save_root, "best_residual.pt")
        save_model(model, save_path, {
            "alpha": args.alpha, "n_samples": args.n_samples,
            "hidden_dim": args.hidden_dim, "latent_dim": args.latent_dim,
            "num_layers": args.num_layers,
            "beta": args.beta, "model_type": "residual",
        })
        metrics = evaluate_residual(model, preprocessor, test_df, args.n_samples, args.alpha, device)
        results = {"CAGPRE_Residual": metrics}
        csv_path = os.path.join(csv_dir, f"train_{args.dataset}.csv")
        save_results_csv(results, csv_path)
        return

    if args.mode == "evaluate":
        if not args.load_path:
            raise ValueError("Provide --load-path for evaluate mode.")
        checkpoint = torch.load(args.load_path, map_location=device, weights_only=False)
        meta = checkpoint.get("meta", {})
        h_dim = meta.get("hidden_dim", args.hidden_dim)
        l_dim = meta.get("latent_dim", args.latent_dim)
        n_layers = meta.get("num_layers", args.num_layers)
        # Auto-detect saved model's context_dim
        saved_input_dim = checkpoint["state_dict"]["encoder_net.0.weight"].shape[1]
        saved_context_dim = saved_input_dim - 1
        if saved_context_dim != c_train.shape[1]:
            print(f"  [WARN] Saved model context_dim={saved_context_dim} != current {c_train.shape[1]}")
            print(f"  Re-fitting preprocessor to match saved model ...")
            cat_cols_compat = ["gender"] if args.ignore_condition else ["gender", "condition"]
            num_cols_compat = ["age"] + extra_context
            preprocessor = TabularPreprocessor(cat_cols=cat_cols_compat, num_cols=num_cols_compat, encoding_type=args.encoding_type)
            preprocessor.fit(train_df)
            c_train = preprocessor.transform_context(train_df)

        is_legacy = "num_layers" not in checkpoint.get("meta", {}) and "num_layers" not in checkpoint
        if is_legacy:
            model = LegacyRobustCAGPRECVAE(context_dim=c_train.shape[1], hidden_dim=h_dim, latent_dim=l_dim, encoding_type=args.encoding_type, cat_dims=preprocessor.get_category_sizes(), embed_dim=args.embed_dim)
        else:
            model = RobustCAGPRECVAE(context_dim=c_train.shape[1], hidden_dim=h_dim, latent_dim=l_dim, encoding_type=args.encoding_type, cat_dims=preprocessor.get_category_sizes(), embed_dim=args.embed_dim, num_layers=n_layers)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        print(f"Loaded model meta: {meta}")
        metrics = evaluate_residual(model, preprocessor, test_df, args.n_samples, args.alpha, device)
        results = {"CAGPRE_Residual_loaded": metrics}
        csv_path = os.path.join(csv_dir, f"evaluate_{args.dataset}.csv")
        save_results_csv(results, csv_path)

    if args.mode == "generate_ranges":
        tuned_path = os.path.join(save_root, "best_residual.pt")
        load_p = args.load_path if args.load_path else tuned_path
        
        h_dim = args.hidden_dim
        l_dim = args.latent_dim
        n_layers = args.num_layers
        
        checkpoint = None
        best_alpha = args.alpha
        if os.path.exists(load_p):
            checkpoint = torch.load(load_p, map_location=device, weights_only=False)
            if "meta" in checkpoint:
                meta = checkpoint["meta"]
                h_dim = meta.get("hidden_dim", args.hidden_dim)
                l_dim = meta.get("latent_dim", args.latent_dim)
                n_layers = meta.get("num_layers", args.num_layers)
                best_alpha = meta.get("alpha", args.alpha)
            else:
                h_dim = checkpoint.get("hidden_dim", args.hidden_dim)
                l_dim = checkpoint.get("latent_dim", args.latent_dim)
                n_layers = checkpoint.get("num_layers", args.num_layers)
                best_alpha = checkpoint.get("alpha", args.alpha)

        is_legacy = False
        if checkpoint is not None:
            is_legacy = "num_layers" not in checkpoint.get("meta", {}) and "num_layers" not in checkpoint
        if is_legacy:
            model = LegacyRobustCAGPRECVAE(context_dim=c_train.shape[1], hidden_dim=h_dim, latent_dim=l_dim, encoding_type=args.encoding_type, cat_dims=preprocessor.get_category_sizes(), embed_dim=args.embed_dim)
        else:
            model = RobustCAGPRECVAE(context_dim=c_train.shape[1], hidden_dim=h_dim, latent_dim=l_dim, encoding_type=args.encoding_type, cat_dims=preprocessor.get_category_sizes(), embed_dim=args.embed_dim, num_layers=n_layers)
        
        if checkpoint is not None:
            print(f"Loading tuned Residual model from {load_p} ...")
            try:
                model.load_state_dict(checkpoint["state_dict"])
                model.to(device)
            except RuntimeError as e:
                print(f"  [WARN] Failed to load old model (architecture mismatch). Retraining from scratch! Error: {str(e)[:100]}...")
                checkpoint = None

        if checkpoint is None:
            print("Training Residual CAGPRE from scratch for generate_ranges ...")
            train_cagpre_residual(
                model, x_train, c_train, x_val, c_val,
                epochs=args.epochs, lr=args.lr, device=device, beta=args.beta,
            )
            save_model(model, tuned_path, {
                "alpha": args.alpha, "n_samples": args.n_samples,
                "hidden_dim": h_dim, "latent_dim": l_dim,
                "beta": args.beta, "model_type": "residual",
                "num_layers": n_layers,
            })
        
        print(f"Generating ranges for full dataset using alpha={best_alpha}...")
        c_full = preprocessor.transform_context(df)
        values = df["value"].values
        pred_low, pred_high, pred_min, pred_max, _ = generate_ranges_and_scores_residual(
            model, c_full, preprocessor, values,
            args.n_samples, best_alpha, device
        )
        
        out_df = df.copy()
        out_df["pred_low"] = pred_low
        out_df["pred_high"] = pred_high
        out_df["pred_min"] = pred_min
        out_df["pred_max"] = pred_max
        
        # 1. Save original per-patient CSV
        cols_to_save = ["person_id", "age", "age_group", "gender", "condition", "value", "pred_low", "pred_high", "pred_min", "pred_max"]
        cols_to_save = [c for c in cols_to_save if c in out_df.columns]
        
        csv_path = os.path.join(csv_dir, f"generated_ranges_{args.dataset}_{variable}.csv")
        out_df[cols_to_save].to_csv(csv_path, index=False)
        print(f"Generated per-patient ranges saved to: {csv_path}")

        # 2. Save grouped summary CSV
        group_cols = ["age_group", "gender"] if getattr(args, "ignore_condition", False) else ["age_group", "gender", "condition"]
        # Ensure only columns that exist in the dataframe are used for grouping
        group_cols = [c for c in group_cols if c in out_df.columns]
        
        summary_df = out_df.groupby(group_cols, observed=True)[["pred_low", "pred_high", "pred_min", "pred_max"]].mean().reset_index()
        summary_csv_path = os.path.join(csv_dir, f"generated_ranges_summary_{args.dataset}_{variable}.csv")
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"Generated grouped summary ranges saved to: {summary_csv_path}")

        return


if __name__ == "__main__":
    main()