import argparse
import torch
import torch.nn as nn

from prepare_data.prepare_data import DataPrepare
from model.stack import Stack
from utils.train_utils import Train, DEVICE
from utils.metrics import MyMetrics


DEFAULT_HYPERPARAMS = {
    "datapath": "./prepare_data/data",
    "datafile": "AU",
    "split_ratio": [0.8, 0.1, 0.1],
    "batch_size": 24,
    "N_EPOCHS": 100,
    "patience": 10,
    "lr": 5e-3,
    "alpha": 1e-1,
    "features": 2,
    "input_seqlen": 96,
    "predict_seqlen": 24,
    "feature_set": 2,
    "num_blocks": 3,
    "encoder_channels": [16, 32, 64, 64],
    "kernel_size": 4,
    "dropout": 0.2,
    "seed": 20,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train TSD-NET for short-term load forecasting.")
    parser.add_argument("--datapath", default=DEFAULT_HYPERPARAMS["datapath"])
    parser.add_argument("--datafile", default=DEFAULT_HYPERPARAMS["datafile"], help="CSV prefix, e.g., AU for AU_data.csv")
    parser.add_argument("--input-len", type=int, default=DEFAULT_HYPERPARAMS["input_seqlen"])
    parser.add_argument("--horizon", type=int, default=DEFAULT_HYPERPARAMS["predict_seqlen"])
    parser.add_argument("--feature-set", type=int, default=DEFAULT_HYPERPARAMS["feature_set"], choices=[1, 2, 3, 4, 5])
    parser.add_argument("--num-blocks", type=int, default=DEFAULT_HYPERPARAMS["num_blocks"])
    parser.add_argument("--epochs", type=int, default=DEFAULT_HYPERPARAMS["N_EPOCHS"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_HYPERPARAMS["batch_size"])
    parser.add_argument("--lr", type=float, default=DEFAULT_HYPERPARAMS["lr"])
    parser.add_argument("--alpha", type=float, default=DEFAULT_HYPERPARAMS["alpha"], help="GDS regularization coefficient")
    parser.add_argument("--no-profiler", action="store_true", help="Disable stage profiler hooks")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hp = dict(DEFAULT_HYPERPARAMS)
    hp.update({
        "datapath": args.datapath,
        "datafile": args.datafile,
        "input_seqlen": args.input_len,
        "predict_seqlen": args.horizon,
        "feature_set": args.feature_set,
        "num_blocks": args.num_blocks,
        "N_EPOCHS": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "alpha": args.alpha,
    })

    dp = DataPrepare(
        datapath=hp["datapath"],
        datafile=hp["datafile"],
        input_steps=hp["input_seqlen"],
        pred_horizon=hp["predict_seqlen"],
        split_ratio=hp["split_ratio"],
        feature_set=hp["feature_set"],
        fit_scaler_on="train",
    )
    hp["features"] = int(dp.num_features)

    model = Stack(
        input_size=hp["features"],
        encoder_channels=hp["encoder_channels"],
        input_seqlen=hp["input_seqlen"],
        forecast_seqlen=hp["predict_seqlen"],
        num_blocks=hp["num_blocks"],
        kernel_size=hp["kernel_size"],
        dropout=hp["dropout"],
        enable_profiler=not args.no_profiler,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=hp["lr"])
    trainer = Train(
        hyperparams=hp,
        model=model,
        optimizer=optimizer,
        loss_func=nn.MSELoss(reduction="mean"),
        data_prepare_cls=dp,
        model_tag=f"TSDNET_D{hp['num_blocks']}_H{hp['predict_seqlen']}_a{hp['alpha']}",
    )

    print(f"Training started on {DEVICE}; alpha={hp['alpha']}; DGR blocks={hp['num_blocks']}")
    trainer.train_model()
    y_hat_std, y_std, *_ = trainer.predict(return_std=True)
    metrics = MyMetrics(y_hat_std, y_std, trainer.dataprepare)
    metrics.print_metrics()


if __name__ == "__main__":
    main()
