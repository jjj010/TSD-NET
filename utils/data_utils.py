
import torch
import numpy as np
import random
import time
from datetime import datetime

def setup_seed(seed: int = 20):
    """
    设置随机种子，保证可复现性
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
    mode: str = 'a'
):
    """
    将超参数、运行时间及最终指标等信息保存到TXT文件。

    :param file_path: TXT文件的路径
    :param hyperparams: 超参数字典
    :param start_time: 训练开始时间 (time.time() 格式)
    :param end_time: 训练结束时间 (time.time() 格式)
    :param train_loss: 训练损失列表
    :param valid_loss: 验证损失列表
    :param metrics: 评估指标，如{"mse":0.1, "mape":..., "smape":...}
    :param mode: 写入模式，默认 'a' 为追加写入，如果想覆盖写就改为 'w'
    """
    elapsed_seconds = end_time - start_time
    elapsed_mins = int(elapsed_seconds // 60)
    elapsed_secs = int(elapsed_seconds % 60)

    with open(file_path, mode, encoding='utf-8') as f:
        f.write("=============== Training Summary ===============\n")
        # 写入当前时间
        f.write(f"Date & Time: {datetime.now()}\n\n")

        # 写入超参数
        f.write("Hyperparameters:\n")
        for k, v in hyperparams.items():
            f.write(f"  {k}: {v}\n")
        f.write("\n")

        # 写入运行时间
        f.write(f"Training time: {elapsed_mins}m {elapsed_secs}s\n\n")

        # 写入Loss情况
        f.write("Losses:\n")
        f.write(f"  - Final Train Loss: {train_loss[-1] if train_loss else 'N/A'}\n")
        f.write(f"  - Final Valid Loss: {valid_loss[-1] if valid_loss else 'N/A'}\n\n")

        # 写入最终评估指标
        f.write("Metrics:\n")
        for key, val in metrics.items():
            f.write(f"  {key}: {val}\n")

        f.write("================================================\n\n")