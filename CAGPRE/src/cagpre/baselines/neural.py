import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple, Union

class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

class VanillaVAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, latent_dim: int = 16):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        h = self.enc(x)
        mu, logvar = self.mu(h), self.logvar(h)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        recon = self.dec(z)
        return recon, mu, logvar

class TransformerAutoencoder(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 32, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(d_model, input_dim)

    def forward(self, x):
        h = self.input_proj(x).unsqueeze(1)  # (B, 1, d_model)
        h = self.encoder(h)
        return self.output_proj(h.squeeze(1))

def train_autoencoder(
    model: nn.Module,
    x_train: np.ndarray, x_val: np.ndarray,
    epochs: int, lr: float, device: str,
    is_vae: bool = False,
    batch_size: int = 1024,
) -> None:
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    n_train = x_train.shape[0]

    # 1. Initialize AMP Scaler
    use_amp = (device != "cpu")
    scaler = torch.amp.GradScaler(enabled=use_amp)

    for epoch in range(epochs):
        model.train()
        perm = np.random.permutation(n_train)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            x_b = torch.tensor(x_train[idx], dtype=torch.float32, device=device)

            optimizer.zero_grad()
            
            # 2. Wrap forward pass and loss in autocast
            with torch.amp.autocast(device_type=device if device != "cpu" else "cpu", enabled=use_amp):
                if is_vae:
                    recon, mu, logvar = model(x_b)
                    recon_loss = mse(recon, x_b)
                    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                    loss = recon_loss + kl_loss
                else:
                    recon = model(x_b)
                    loss = mse(recon, x_b)

            if torch.isnan(loss):
                continue

            # 3. Use Scaler for backward pass
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            n_batches += 1

    if n_batches > 0:
        print(f"    Final loss: {epoch_loss / n_batches:.4f}")

def reconstruction_errors(
    model: nn.Module, data: np.ndarray, device: str, is_vae: bool = False,
    batch_size: int = 2048,
) -> np.ndarray:
    model.eval()
    errors = []
    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            x_b = torch.tensor(data[start:start + batch_size], dtype=torch.float32, device=device)
            if is_vae:
                recon, _, _ = model(x_b)
            else:
                recon = model(x_b)
            err = ((recon - x_b) ** 2).mean(dim=1).cpu().numpy()
            errors.append(err)
    return np.concatenate(errors)

def evaluate_autoencoder_model(
    model: nn.Module,
    x_train: np.ndarray, x_val: np.ndarray, x_test: np.ndarray,
    y_test: np.ndarray, epochs: int, lr: float, device: str,
    is_vae: bool = False,
) -> Dict[str, float]:
    train_autoencoder(model, x_train, x_val, epochs, lr, device, is_vae=is_vae)
    train_err = reconstruction_errors(model, x_train, device, is_vae=is_vae)
    threshold = np.quantile(train_err, 0.95)
    test_err = reconstruction_errors(model, x_test, device, is_vae=is_vae)
    y_pred = (test_err > threshold).astype(int)
    from ..engines.shared import compute_metrics
    return compute_metrics(y_test, y_pred)

