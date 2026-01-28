import os
import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.data_utils import setup_seed, save_statistics_to_txt
from utils.metrics import MyMetrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path = os.getcwd()


class Train:
    def __init__(self, hyperparams, model, optimizer, loss_func, data_prepare_cls, model_tag="default"):
        """
        Args:
            hyperparams: Dictionary of hyperparameters.
            model: PyTorch model instance.
            optimizer: PyTorch optimizer.
            loss_func: Loss function for the main forecasting objective.
            data_prepare_cls: DataPrepare class or an instance.
            model_tag: Tag used to distinguish different runs/modules.
        """
        self.model_tag = model_tag
        self.datapath = hyperparams["datapath"]
        self.datafile = hyperparams["datafile"]
        self.split_ratio = hyperparams["split_ratio"]
        self.N_EPOCHS = hyperparams["N_EPOCHS"]
        self.lr = hyperparams["lr"]
        self.batch_size = hyperparams["batch_size"]
        self.input_seqlen = hyperparams["input_seqlen"]
        self.predict_seqlen = hyperparams["predict_seqlen"]
        self.features = hyperparams["features"]

        # Initialize data preparation
        self.dataprepare = self._init_dataprepare(data_prepare_cls)

        self.train_generator = None
        self.valid_generator = None
        self.test_generator = None

        self.model = model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.ar_loss_func = nn.MSELoss()

        # Loss history
        self.train_loss = []
        self.valid_loss = []
        self.ar_loss_history = []
        self.decoder_loss_history = []

        self.mse = 0

        setup_seed(20)
        self._prepare_data()

    def _init_dataprepare(self, data_prepare_cls):
        # If an instance is passed in, use it directly
        if hasattr(data_prepare_cls, "prepare_data"):
            return data_prepare_cls

        # Otherwise, try to instantiate it (handle possible argument name differences)
        try:
            return data_prepare_cls(
                datapath=self.datapath,
                datafile=self.datafile,
                input_steps=self.input_seqlen,
                pred_horizon=self.predict_seqlen,
                split_ratio=self.split_ratio,
            )
        except TypeError:
            # Fallback to legacy keyword if needed
            return data_prepare_cls(
                datapath=self.datapath,
                datafile=self.datafile,
                input_steps=self.input_seqlen,
                pred_horizion=self.predict_seqlen,  # legacy typo in some codebases
                split_ratio=self.split_ratio,
            )

    def _prepare_data(self):
        tvt_data = self.dataprepare.prepare_data()
        from prepare_data.prepare_data import Datasets

        train_dataset = Datasets(tvt_data[0], tvt_data[1])
        valid_dataset = Datasets(tvt_data[2], tvt_data[3])
        test_dataset = Datasets(tvt_data[4], tvt_data[5])

        self.train_generator = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
        self.valid_generator = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_generator = DataLoader(test_dataset, batch_size=2048, shuffle=False)

    def save_state(self, state, filepath=path + "/trained_file"):
        """
        Save model checkpoint under:
          <filepath>/<datafile>/<datafile>-<input>to<predict>-<model_tag>.pt
        """
        os.makedirs(filepath, exist_ok=True)
        dataset_folder = os.path.join(filepath, self.datafile)
        os.makedirs(dataset_folder, exist_ok=True)

        filename = os.path.join(
            dataset_folder,
            f"{self.datafile}-{self.input_seqlen}to{self.predict_seqlen}-{self.model_tag}.pt",
        )
        torch.save(state, filename)
        print(f"Model saved to: {filename}")

    def load_state(self, filepath=path + "/trained_file"):
        print("Loading model and optimizer state...")

        dataset_folder = os.path.join(filepath, self.datafile)
        filename = os.path.join(
            dataset_folder,
            f"{self.datafile}-{self.input_seqlen}to{self.predict_seqlen}-{self.model_tag}.pt",
        )

        checkpoint = torch.load(filename, map_location=DEVICE)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.train_loss = checkpoint.get("train_loss", [])
        self.valid_loss = checkpoint.get("valid_loss", [])
        self.mse = checkpoint.get("mse", 0)
        self.dataprepare = checkpoint.get("dataprepare", self.dataprepare)

        print(f"Model loaded from: {filename}")

    def train_ar_model(self):
        """Train the autoregressive (AR) component."""
        self.model.train()
        profiler = getattr(self.model, "profiler", None)

        for epoch in range(self.N_EPOCHS):
            epoch_loss = 0.0
            for x, y in self.train_generator:
                x, y = x.to(DEVICE), y.to(DEVICE)

                if profiler:
                    profiler.batch_begin()

                self.optimizer.zero_grad(set_to_none=True)

                if profiler:
                    with profiler.fwd():
                        _, _, ar_predictions, _ = self.model(x)
                else:
                    _, _, ar_predictions, _ = self.model(x)

                ar_loss = self.ar_loss_func(
                    ar_predictions.squeeze(2),
                    y[:, : ar_predictions.shape[1]],
                )

                if profiler:
                    with profiler.bwd():
                        ar_loss.backward()
                else:
                    ar_loss.backward()

                if profiler:
                    with profiler.opt():
                        self.optimizer.step()
                    profiler.batch_end()
                else:
                    self.optimizer.step()

                epoch_loss += float(ar_loss.item())

            avg_epoch_loss = epoch_loss / max(1, len(self.train_generator))
            self.ar_loss_history.append(avg_epoch_loss)
            self.train_loss.append(avg_epoch_loss)

            valid_loss = self._evaluate(self.valid_generator)
            self.valid_loss.append(valid_loss)

            print(
                f"AR Epoch {epoch + 1} | AR_Loss {avg_epoch_loss:.4f} | Val_Loss {valid_loss:.4f}"
            )

        if profiler:
            results_dir = os.path.join(os.getcwd(), "results")
            os.makedirs(results_dir, exist_ok=True)
            csv_path = os.path.join(results_dir, f"train_phases_AR_{self.model_tag}.csv")
            df_phase = profiler.summarize_training_phases(save_csv_path=csv_path)
            print(df_phase.to_string(index=False))
            print(f"[Profiler] AR phase report saved to: {csv_path}")

    def train_decoder_model(self):
        """Train the decoder (main forecasting network)."""
        self.model.train()
        profiler = getattr(self.model, "profiler", None)

        # Freeze AR layers during decoder training
        if hasattr(self.model, "block_first") and hasattr(self.model.block_first, "ar_layer"):
            for p in self.model.block_first.ar_layer.parameters():
                p.requires_grad = False
        if hasattr(self.model, "blocks_rest"):
            for block in self.model.blocks_rest:
                if hasattr(block, "ar_layer"):
                    for p in block.ar_layer.parameters():
                        p.requires_grad = False

        for epoch in range(self.N_EPOCHS):
            epoch_decoder_loss = 0.0
            epoch_residual_loss = 0.0

            for x, y in self.train_generator:
                x, y = x.to(DEVICE), y.to(DEVICE)

                if profiler:
                    profiler.batch_begin()

                self.optimizer.zero_grad(set_to_none=True)

                if profiler:
                    with profiler.fwd():
                        forecast_output, residual, _, _ = self.model(x)
                else:
                    forecast_output, residual, _, _ = self.model(x)

                decoder_loss = self.loss_func(forecast_output.squeeze(2), y)
                residual_loss = torch.mean(torch.abs(residual))
                total_loss = decoder_loss + residual_loss

                if profiler:
                    with profiler.bwd():
                        total_loss.backward()
                else:
                    total_loss.backward()

                if profiler:
                    with profiler.opt():
                        self.optimizer.step()
                    profiler.batch_end()
                else:
                    self.optimizer.step()

                epoch_decoder_loss += float(decoder_loss.item())
                epoch_residual_loss += float(residual_loss.item())

            avg_decoder_loss = epoch_decoder_loss / max(1, len(self.train_generator))
            avg_residual_loss = epoch_residual_loss / max(1, len(self.train_generator))
            self.decoder_loss_history.append(avg_decoder_loss)

            print(
                f"Decoder Epoch {epoch + 1} | Decoder_Loss {avg_decoder_loss:.4f} | Residual_Loss {avg_residual_loss:.4f}"
            )

        if profiler:
            results_dir = os.path.join(os.getcwd(), "results")
            os.makedirs(results_dir, exist_ok=True)
            csv_path = os.path.join(results_dir, f"train_phases_Decoder_{self.model_tag}.csv")
            df_phase = profiler.summarize_training_phases(save_csv_path=csv_path)
            print(df_phase.to_string(index=False))
            print(f"[Profiler] Decoder phase report saved to: {csv_path}")

    def train_model(self, txt_file_path="./training_logs.txt"):
        start_time = time.time()

        print(f"DEVICE: {DEVICE}")
        print("Training AR model...")
        self.train_ar_model()

        print("Training decoder model...")
        self.train_decoder_model()

        # Save final checkpoint
        my_state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "train_loss": self.train_loss,
            "valid_loss": self.valid_loss,
            "dataprepare": self.dataprepare,
            "mse": self.mse,
            "ar_loss_history": self.ar_loss_history,
            "decoder_loss_history": self.decoder_loss_history,
        }
        self.save_state(state=my_state)

        y_hat, y, ar_predictions, residual, encoder_inputs = self.predict()
        my_metrics = MyMetrics(y_pred=y_hat, y_true=y, un_std=self.dataprepare)

        metrics_dict = {
            "last_ar_loss": self.ar_loss_history[-1] if self.ar_loss_history else None,
            "last_decoder_loss": self.decoder_loss_history[-1] if self.decoder_loss_history else None,
            "mse": my_metrics.metrics["mse"],
            "mape": my_metrics.metrics["mape"],
            "smape": my_metrics.metrics["smape"],
            "r2": my_metrics.metrics["r2"],
        }

        end_time = time.time()

        save_statistics_to_txt(
            file_path=txt_file_path,
            hyperparams={
                "datapath": self.datapath,
                "datafile": self.datafile,
                "split_ratio": self.split_ratio,
                "N_EPOCHS": self.N_EPOCHS,
                "lr": self.lr,
                "batch_size": self.batch_size,
                "input_seqlen": self.input_seqlen,
                "predict_seqlen": self.predict_seqlen,
                "features": self.features,
            },
            start_time=start_time,
            end_time=end_time,
            train_loss=self.train_loss,
            valid_loss=self.valid_loss,
            metrics=metrics_dict,
            mode="a",
        )

    def _evaluate(self, data_loader):
        """Compute loss over a given dataloader."""
        self.model.eval()
        profiler = getattr(self.model, "profiler", None)

        epoch_loss = 0.0
        with torch.no_grad():
            for x, y in data_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                if profiler:
                    profiler.batch_begin()

                forecast_output, residual, _, _ = self.model(x)

                if profiler:
                    profiler.batch_end()

                loss = self.loss_func(forecast_output.squeeze(2), y) + torch.mean(torch.abs(residual))
                epoch_loss += float(loss.item())

        return epoch_loss / max(1, len(data_loader))

    def predict(self):
        """Run inference on the test set and return de-normalized outputs."""
        self.model.eval()
        all_y_hat = []
        all_y = []
        all_ar_predictions = []
        all_residuals = []
        all_encoder_inputs = []

        if DEVICE.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device=DEVICE)

        with torch.no_grad():
            for x, y in self.test_generator:
                x, y = x.to(DEVICE), y.to(DEVICE)
                y_hat, residual, ar_predictions, encoder_inputs = self.model(x)

                all_y_hat.append(y_hat.squeeze(2).cpu().numpy())
                all_y.append(y.cpu().numpy())
                all_ar_predictions.append(ar_predictions.squeeze(2).cpu().numpy())
                all_residuals.append(residual.squeeze(2).cpu().numpy())
                all_encoder_inputs.extend([ei.cpu().numpy() for ei in encoder_inputs])

        # Optional: export per-stage profiling report (if profiler exists)
        profiler = getattr(self.model, "profiler", None)
        if profiler is not None:
            results_dir = os.path.join(os.getcwd(), "results")
            os.makedirs(results_dir, exist_ok=True)
            csv_path = os.path.join(results_dir, f"per_stage_efficiency_{self.model_tag}.csv")
            df_prof = profiler.summarize(save_csv_path=csv_path)
            print("\n[Per-Stage Efficiency]")
            print(df_prof.to_string(index=False))
            print(f"[Profiler] Report saved to: {csv_path}")

        y_hat_unnorm = self.dataprepare.un_standardize(np.concatenate(all_y_hat))
        y_unnorm = self.dataprepare.un_standardize(np.concatenate(all_y))
        ar_pred_unnorm = self.dataprepare.un_standardize(np.concatenate(all_ar_predictions))
        residual_unnorm = self.dataprepare.un_standardize(np.concatenate(all_residuals))

        df_out = pd.DataFrame(
            {
                "Actual": y_unnorm.flatten(),
                "Predicted": y_hat_unnorm.flatten(),
            }
        )
        filename = f"prediction_results_{self.model_tag}.xlsx"
        df_out.to_excel(filename, index=False)
        print(f"Prediction results saved to: {filename}")

        return y_hat_unnorm, y_unnorm, ar_pred_unnorm, residual_unnorm, all_encoder_inputs
