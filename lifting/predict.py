import torch
import numpy as np

from lifting.models.fcn import FCN
from lifting.models.gcn import GCN


def load_model(checkpoint_path, device=None):
    """
    Load a saved lifting model from a checkpoint.

    Args:
        checkpoint_path: path to .pth file saved by train.py
        device: 'cpu', 'cuda', or None (auto-detect)

    Returns:
        model:  loaded model in eval mode
        ckpt:   full checkpoint dict (contains normalization stats)
        device: resolved device string
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_type = ckpt.get('model_type', 'fcn')
    hidden_dim = ckpt.get('hidden_dim', 1024)

    if model_type == 'fcn':
        model = FCN(hidden_dim=hidden_dim)
    elif model_type == 'gcn':
        model = GCN(hidden_dim=hidden_dim)
    else:
        raise ValueError(f"Unknown model_type in checkpoint: {model_type!r}")

    model.load_state_dict(ckpt['model_state'])
    model.to(device).eval()

    return model, ckpt, device


def predict(model, keypoints_2d, ckpt, device):
    """
    Lift 2D keypoints to 3D coordinates.

    Args:
        model:        trained lifting model (from load_model)
        keypoints_2d: np.array (N, 17, 2) or (17, 2) for a single pose
        ckpt:         checkpoint dict with normalization stats
        device:       torch device string

    Returns:
        keypoints_3d: np.array (N, 17, 3) or (17, 3) for a single pose
    """
    single = keypoints_2d.ndim == 2
    if single:
        keypoints_2d = keypoints_2d[np.newaxis]   # (1, 17, 2)

    x = keypoints_2d.reshape(len(keypoints_2d), -1).astype(np.float32)
    x = (x - ckpt['mean_2d']) / ckpt['std_2d']

    with torch.no_grad():
        pred = model(torch.tensor(x, device=device)).cpu().numpy()

    # Denormalize
    pred = pred * ckpt['std_3d'] + ckpt['mean_3d']
    pred = pred.reshape(-1, 17, 3)

    return pred[0] if single else pred
