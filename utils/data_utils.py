import torch
import numpy as np
import random
from datetime import datetime


def setup_seed(seed: int = 20) -> None:
    """
    Set random seeds for reproducibility across PyTorch, NumPy, and Python.

    Args:
        seed: Random seed value.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True


def save_statistics_to_txt(
    file_path: str,
    hyperparams: dict,
    start_time: float,
    end_time: float,
    train_loss: list,
    valid_loss: list,
    metrics: dict,
    mode: str = "a",
) -> None:
    """
    Save training configuration, runtime, losses, and evaluation metrics to a TXT file.

    Args:
        file_path: Path to the output TXT file.
        hyperparams: Dictionary of hyperparameters.
        start_time: Training start timestamp (time.time()).
        end_time: Training end timestamp (time.time()).
        train_loss: List of training losses.
        valid_loss: List of validation losses.
        metrics: Dictionary of evaluation metrics (e.g., mse, mape, r2).
        mode: File write mode ("a" for append, "w" for overwrite).
    """
    elapsed_seconds = end_time - start_time
    elapsed_mins = int(elapsed_seconds // 60)
    elapsed_secs = int(elapsed_seconds % 60)

    with open(file_path, mode, encoding="utf-8") as f:
        f.write("=============== Training Summary ===============\n")
        f.write(f"Date & Time: {datetime.now()}\n\n")

        f.write("Hyperparameters:\n")
        for k, v in hyperparams.items():
            f.write(f"  {k}: {v}\n")
        f.write("\n")

        f.write(f"Training time: {elapsed_mins}m {elapsed_secs}s\n\n")

        f.write("Losses:\n")
        f.write(f"  Final Train Loss: {train_loss[-1] if train_loss else 'N/A'}\n")
        f.write(f"  Final Valid Loss: {valid_loss[-1] if valid_loss else 'N/A'}\n\n")

        f.write("Metrics:\n")
        for key, val in metrics.items():
            f.write(f"  {key}: {val}\n")

        f.write("================================================\n\n")
