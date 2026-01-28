import torch
import torch.nn as nn

from prepare_data.prepare_data import DataPrepare
from model.stack import Stack
from utils.train_utils import Train
from utils.metrics import MyMetrics

# -----------------------------
# Hyperparameters
# -----------------------------
HyperParams = {
    "datapath": "./prepare_data/data",
    "datafile": "AD",
    "split_ratio": [0.8, 0.1, 0.1],
    "batch_size": 24,
    "N_EPOCHS": 100,
    "patience": 10,
    "lr": 5e-3,
    "features": 2,
    "input_seqlen": 96,
    "predict_seqlen": 24,
    "forecast_type": "transformer",
    "use_ar": False,
    "use_estimate": False,
    "lambda_recon": 0.1,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)


def main() -> None:
    load_checkpoint = False

    dp = DataPrepare(
        datapath=HyperParams["datapath"],
        datafile=HyperParams["datafile"],
        input_steps=HyperParams["input_seqlen"],
        pred_horizon=HyperParams["predict_seqlen"],
        split_ratio=HyperParams["split_ratio"],
    )
    HyperParams["features"] = int(dp.num_features)

    model = Stack(
        input_size=HyperParams["features"],
        encoder_channels=[16, 32, 64, 64],
        input_seqlen=HyperParams["input_seqlen"],
        forecast_seqlen=HyperParams["predict_seqlen"],
        forecast_type=HyperParams["forecast_type"],
        use_ar=HyperParams["use_ar"],
        use_estimate=HyperParams["use_estimate"],
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=HyperParams["lr"])
    loss_func = nn.MSELoss(reduction="mean")

    base_tag = "block3f124"
    model_tag = (
        f"{base_tag}"
        f'__ft-{HyperParams["forecast_type"]}'
        f'__ar-{int(HyperParams["use_ar"])}'
        f'__est-{int(HyperParams["use_estimate"])}'
    )

    trainer = Train(
        hyperparams=HyperParams,
        model=model,
        optimizer=optimizer,
        loss_func=loss_func,
        data_prepare_cls=dp,
        model_tag=model_tag,
    )

    if load_checkpoint:
        trainer.load_state()

    print(f"Training started (device={DEVICE}, patience={HyperParams['patience']})...")
    trainer.train_model()

    y_hat_std, y_std, *_ = trainer.predict(return_std=True)
    metrics = MyMetrics(y_hat_std, y_std, trainer.dataprepare)
    metrics.print_metrics()


if __name__ == "__main__":
    main()
