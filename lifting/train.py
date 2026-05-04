import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm

from lifting.models.fcn import FCN
from lifting.models.gcn import GCN
from data.loader import PoseDataset, load_human36m, generate_synthetic_data
from data.preprocess import normalize_2d, normalize_3d


def train(
    model_type='fcn',
    data_path=None,
    epochs=50,
    batch_size=1024,
    lr=1e-3,
    hidden_dim=1024,
    save_path='checkpoints',
    device=None,
):
    """
    Train the lifting model.

    Args:
        model_type: 'fcn' or 'gcn'
        data_path:  path to directory with Human3.6M train.npz / test.npz;
                    falls back to synthetic data if None or path not found
        epochs:     number of training epochs
        batch_size: mini-batch size
        lr:         initial learning rate
        hidden_dim: hidden layer width (FCN) or channels (GCN)
        save_path:  directory to write the best checkpoint
        device:     'cpu', 'cuda', or None (auto-detect)

    Returns:
        Path to the saved checkpoint file
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    if data_path and Path(data_path).exists():
        print("Loading Human3.6M data ...")
        train_2d, train_3d, val_2d, val_3d = load_human36m(data_path)
    else:
        print("Human3.6M not found — using synthetic data for demonstration.")
        train_2d, train_3d, val_2d, val_3d = generate_synthetic_data()

    # Flatten joints × coords
    train_2d = train_2d.reshape(len(train_2d), -1)
    val_2d   = val_2d.reshape(len(val_2d), -1)
    train_3d = train_3d.reshape(len(train_3d), -1)
    val_3d   = val_3d.reshape(len(val_3d), -1)

    # Normalize
    train_2d, mean_2d, std_2d = normalize_2d(train_2d)
    val_2d   = (val_2d - mean_2d) / std_2d
    train_3d, mean_3d, std_3d = normalize_3d(train_3d)
    val_3d   = (val_3d - mean_3d) / std_3d

    train_loader = DataLoader(PoseDataset(train_2d, train_3d), batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(PoseDataset(val_2d, val_3d),     batch_size=batch_size, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────────
    if model_type == 'fcn':
        model = FCN(hidden_dim=hidden_dim).to(device)
    elif model_type == 'gcn':
        model = GCN(hidden_dim=min(hidden_dim, 256)).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}. Choose 'fcn' or 'gcn'.")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.MSELoss()

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_path / f'{model_type}_best.pth'

    best_val_loss = float('inf')

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(x)
        train_loss /= len(train_loader.dataset)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_loss += criterion(model(x), y).item() * len(x)
        val_loss /= len(val_loader.dataset)

        scheduler.step()
        print(f"Epoch {epoch:3d} | train_loss={train_loss:.5f}  val_loss={val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_type': model_type,
                'hidden_dim': hidden_dim if model_type == 'fcn' else min(hidden_dim, 256),
                'model_state': model.state_dict(),
                'mean_2d': mean_2d,
                'std_2d':  std_2d,
                'mean_3d': mean_3d,
                'std_3d':  std_3d,
            }, ckpt_path)

    print(f"\nTraining complete. Best val_loss={best_val_loss:.5f}")
    print(f"Checkpoint saved to: {ckpt_path}")
    return ckpt_path
