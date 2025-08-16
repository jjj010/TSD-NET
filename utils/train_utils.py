
import torch
import torch.nn as nn
import torch.utils.data
import time
import numpy as np
import os
from torch.utils.data import DataLoader
from utils.data_utils import setup_seed, save_statistics_to_txt
from utils.metrics import MyMetrics
import pandas as pd
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path = os.getcwd()


class Train:
    def __init__(self, hyperparams, model, optimizer, loss_func, data_prepare_cls,model_tag='default'):
        """
        :param hyperparams: 包含超参数的字典
        :param model: PyTorch 模型实例
        :param optimizer: PyTorch 优化器
        :param loss_func: 损失函数
        :param data_prepare_cls: 需要你传入的 prepare_data.DataPrepare 类或其实例
        """
        self.model_tag = model_tag  # 区分不同基本块
        self.datapath = hyperparams['datapath']
        self.datafile = hyperparams['datafile']
        self.split_ratio = hyperparams['split_ratio']
        self.N_EPOCHS = hyperparams['N_EPOCHS']
        self.lr = hyperparams['lr']
        self.batch_size = hyperparams['batch_size']
        self.input_seqlen = hyperparams['input_seqlen']
        self.predict_seqlen = hyperparams['predict_seqlen']
        self.features = hyperparams['features']

        # 初始化数据处理对象
        self.dataprepare = data_prepare_cls(
            datapath=self.datapath,
            datafile=self.datafile,
            input_steps=self.input_seqlen,
            pred_horizion=self.predict_seqlen,
            split_ratio=self.split_ratio
        )

        self.train_generator = None
        self.valid_generator = None
        self.test_generator = None
        self.model = model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.ar_loss_func = nn.MSELoss()

        # 初始化损失历史列表
        self.train_loss = []
        self.valid_loss = []
        self.ar_loss_history = []
        self.decoder_loss_history = []

        # 一个简单的 mse 值记录，可视需求使用
        self.mse = 0

        setup_seed(20)
        self._prepare_data()


    def _prepare_data(self):
        # 从数据准备类获取 训练/验证/测试 数据
        tvt_data = self.dataprepare.prepare_data()

        # 这里从 prepare_data.prepare_data 导入 Datasets
        from prepare_data.prepare_data import Datasets

        train_dataset = Datasets(tvt_data[0], tvt_data[1])
        valid_dataset = Datasets(tvt_data[2], tvt_data[3])
        test_dataset  = Datasets(tvt_data[4], tvt_data[5])

        self.train_generator = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
        self.valid_generator = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_generator  = DataLoader(test_dataset,  batch_size=2048,       shuffle=False)

    # def save_state(self, state, filepath=path + "/trained_file"):
    #     """
    #     将模型及其相关状态保存到
    #       <filepath>/<self.datafile>/<self.datafile>-{input_seqlen}to{predict_seqlen}
    #     """
    #     if not os.path.exists(filepath):
    #         os.makedirs(filepath, exist_ok=True)
    #
    #     # 根据数据集名称，创建一个子文件夹
    #     dataset_folder = os.path.join(filepath, self.datafile)
    #     if not os.path.exists(dataset_folder):
    #         os.makedirs(dataset_folder, exist_ok=True)
    #
    #     # 拼接最终的文件路径
    #     filename = os.path.join(
    #         dataset_folder,
    #         f'{self.datafile}-{self.input_seqlen}to{self.predict_seqlen}'
    #     )
    #
    #     torch.save(state, filename)
    #     print(f"Model saved to: {filename}")
    def save_state(self, state, filepath=path + "/trained_file"):
        """
        将模型及其相关状态保存到指定路径，增加 model_tag 用于区分不同模块（如 block1、block2）
        """
        if not os.path.exists(filepath):
            os.makedirs(filepath, exist_ok=True)

        dataset_folder = os.path.join(filepath, self.datafile)
        if not os.path.exists(dataset_folder):
            os.makedirs(dataset_folder, exist_ok=True)

        # 使用 model_tag 作为文件名的一部分
        filename = os.path.join(
            dataset_folder,
            f'{self.datafile}-{self.input_seqlen}to{self.predict_seqlen}-{self.model_tag}.pt'
        )

        torch.save(state, filename)
        print(f"Model saved to: {filename}")

    # def load_state(self, filepath=path + "/trained_file"):
    #     """
    #     从
    #       <filepath>/<self.datafile>/<self.datafile>-{input_seqlen}to{predict_seqlen}
    #     加载已有的模型、优化器等状态
    #     """
    #     print("Loading model and optimizer state")
    #
    #     # 构造子文件夹
    #     dataset_folder = os.path.join(filepath, self.datafile)
    #     # 拼接文件名
    #     filename = os.path.join(
    #         dataset_folder,
    #         f'{self.datafile}-{self.input_seqlen}to{self.predict_seqlen}'
    #     )
    #     checkpoint = torch.load(filename)
    #
    #     self.model.load_state_dict(checkpoint['model'])
    #     self.optimizer.load_state_dict(checkpoint['optimizer'])
    #     self.train_loss = checkpoint['train_loss']
    #     self.valid_loss = checkpoint['valid_loss']
    #     self.mse = checkpoint['mse']
    #     self.dataprepare = checkpoint['dataprepare']
    #
    #     print(f"Model loaded from: {filename}")
    def load_state(self, filepath=path + "/trained_file"):
        print("Loading model and optimizer state")

        dataset_folder = os.path.join(filepath, self.datafile)
        filename = os.path.join(
            dataset_folder,
            f'{self.datafile}-{self.input_seqlen}to{self.predict_seqlen}-{self.model_tag}.pt'
        )

        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.train_loss = checkpoint['train_loss']
        self.valid_loss = checkpoint['valid_loss']
        self.mse = checkpoint['mse']
        self.dataprepare = checkpoint['dataprepare']

        print(f"Model loaded from: {filename}")

    def train_ar_model(self):
        """训练自回归部分 (AR Model)"""
        self.model.train()
        profiler = getattr(self.model, "profiler", None)

        for epoch in range(self.N_EPOCHS):
            epoch_loss = 0
            for i, (x, y) in enumerate(self.train_generator):
                x, y = x.to(DEVICE), y.to(DEVICE)

                if profiler: profiler.batch_begin()
                self.optimizer.zero_grad(set_to_none=True)

                # ===== forward =====
                if profiler:
                    with profiler.fwd():
                        _, _, ar_predictions, _ = self.model(x)
                else:
                    _, _, ar_predictions, _ = self.model(x)

                ar_loss = self.ar_loss_func(
                    ar_predictions.squeeze(2),
                    y[:, :ar_predictions.shape[1]]
                )

                # ===== backward =====
                if profiler:
                    with profiler.bwd():
                        ar_loss.backward()
                else:
                    ar_loss.backward()

                # ===== optim =====
                if profiler:
                    with profiler.opt():
                        self.optimizer.step()
                    profiler.batch_end()
                else:
                    self.optimizer.step()

                epoch_loss += ar_loss.item()

            avg_epoch_loss = epoch_loss / len(self.train_generator)
            self.ar_loss_history.append(avg_epoch_loss)
            self.train_loss.append(avg_epoch_loss)

            valid_loss = self._evaluate(self.valid_generator)
            self.valid_loss.append(valid_loss)

            print(f'AR Model Epoch: {epoch + 1} | AR_Loss: {avg_epoch_loss:.4f} | Valid_Loss: {valid_loss:.4f}')

        # —— 阶段结束：导出训练三段统计（batch 粒度）——
        if profiler:
            import os
            results_dir = os.path.join(os.getcwd(), "results")
            os.makedirs(results_dir, exist_ok=True)
            csv_path = os.path.join(results_dir, f"train_phases_AR_{self.model_tag}.csv")
            df_phase = profiler.summarize_training_phases(save_csv_path=csv_path)
            print(df_phase.to_string(index=False))
            print(f"[Profiler] AR phases saved to: {csv_path}")

    def train_decoder_model(self):

        self.model.train()
        profiler = getattr(self.model, "profiler", None)


        for param in self.model.block_first.ar_layer.parameters():
            param.requires_grad = False
        for block in self.model.blocks_rest:
            for param in block.ar_layer.parameters():
                param.requires_grad = False

        for epoch in range(self.N_EPOCHS):
            epoch_decoder_loss = 0
            epoch_residual_loss = 0

            for i, (x, y) in enumerate(self.train_generator):
                x, y = x.to(DEVICE), y.to(DEVICE)

                if profiler: profiler.batch_begin()
                self.optimizer.zero_grad(set_to_none=True)

                # ===== forward =====
                if profiler:
                    with profiler.fwd():
                        forecast_output, residual, _, _ = self.model(x)
                else:
                    forecast_output, residual, _, _ = self.model(x)

                decoder_loss = self.loss_func(forecast_output.squeeze(dim=2), y)
                residual_loss = torch.mean(torch.abs(residual))
                total_loss = decoder_loss + residual_loss


                if profiler:
                    with profiler.bwd():
                        total_loss.backward()
                else:
                    total_loss.backward()

                # ===== optim =====
                if profiler:
                    with profiler.opt():
                        self.optimizer.step()
                    profiler.batch_end()
                else:
                    self.optimizer.step()

                epoch_decoder_loss += decoder_loss.item()
                epoch_residual_loss += residual_loss.item()

            avg_decoder_loss = epoch_decoder_loss / len(self.train_generator)
            avg_residual_loss = epoch_residual_loss / len(self.train_generator)
            self.decoder_loss_history.append(avg_decoder_loss)

            print(
                f'Decoder Model Epoch: {epoch + 1} | Decoder_Loss: {avg_decoder_loss:.4f} | Residual_Loss: {avg_residual_loss:.4f}'
            )


        if profiler:
            import os
            results_dir = os.path.join(os.getcwd(), "results")
            os.makedirs(results_dir, exist_ok=True)
            csv_path = os.path.join(results_dir, f"train_phases_Decoder_{self.model_tag}.csv")
            df_phase = profiler.summarize_training_phases(save_csv_path=csv_path)
            print(df_phase.to_string(index=False))
            print(f"[Profiler] Decoder phases saved to: {csv_path}")

    def train_model(self, txt_file_path='./training_logs.txt'):

        start_time = time.time()

        print(f'DEVICE: {DEVICE}')
        print('Training AR Model...')
        self.train_ar_model()

        print('Training Decoder Model...')
        self.train_decoder_model()

        # 保存最终模型
        my_state = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'train_loss': self.train_loss,
            'valid_loss': self.valid_loss,
            'dataprepare': self.dataprepare,
            'mse': self.mse,
            'ar_loss_history': self.ar_loss_history,
            'decoder_loss_history': self.decoder_loss_history,
        }
        self.save_state(state=my_state)


        y_hat, y, ar_predictions, residual, encoder_inputs = self.predict()

        my_metrics = MyMetrics(y_pred=y_hat, y_true=y, un_std=self.dataprepare)

        metrics_dict = {
            "last_ar_loss": self.ar_loss_history[-1] if self.ar_loss_history else None,
            "last_decoder_loss": self.decoder_loss_history[-1] if self.decoder_loss_history else None,
            "mse": my_metrics.metrics['mse'],
            "mape": my_metrics.metrics['mape'],
            "smape": my_metrics.metrics['smape'],
            "r2":my_metrics.metrics['r2'] }

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
                "features": self.features
            },
            start_time=start_time,
            end_time=end_time,
            train_loss=self.train_loss,
            valid_loss=self.valid_loss,
            metrics=metrics_dict,
            mode='a'  # 'a' 是追加写入，可改成 'w' 覆盖写
        )

    def _evaluate(self, data_loader):
        """在给定 data_loader 上计算一次预测损失"""
        self.model.eval()
        profiler = getattr(self.model, "profiler", None)
        epoch_loss = 0
        with torch.no_grad():
            for i, (x, y) in enumerate(data_loader):
                x, y = x.to(DEVICE), y.to(DEVICE)
                if profiler: profiler.batch_begin()
                forecast_output, residual, _, _ = self.model(x)
                if profiler: profiler.batch_end()
                loss = self.loss_func(forecast_output.squeeze(dim=2), y) + torch.mean(torch.abs(residual))
                epoch_loss += loss.item()
        return epoch_loss / len(data_loader)

    # def predict(self):
    #     """
    #     在测试集上做一次预测，并返回 y_hat, y, ar_predictions, residual, encoder_inputs
    #     其中 encoder_inputs 是个 list，包含每个 block 的输入信息
    #     """
    #     self.model.eval()
    #     with torch.no_grad():
    #         for i, (x, y) in enumerate(self.test_generator):
    #             x, y = x.to(DEVICE), y.to(DEVICE)
    #             y_hat, residual, ar_predictions, encoder_inputs = self.model(x)
    #
    #             y_hat = y_hat.squeeze(dim=2)
    #             ar_predictions = ar_predictions.squeeze(dim=2)
    #             residual = residual.squeeze(dim=2)
    #             break  # 只拿一个 batch 演示
    #     return (
    #         y_hat.cpu().numpy(),
    #         y.cpu().numpy(),
    #         ar_predictions.cpu().numpy(),
    #         residual.cpu().numpy(),
    #         [encoder_input.cpu().numpy() for encoder_input in encoder_inputs]
    #     )

    # def predict(self):
    #     self.model.eval()
    #     all_y_hat = []
    #     all_y = []
    #     all_ar_predictions = []
    #     all_residuals = []
    #     all_encoder_inputs = []
    #
    #     with torch.no_grad():
    #         for i, (x, y) in enumerate(self.test_generator):
    #             x, y = x.to(DEVICE), y.to(DEVICE)
    #             y_hat, residual, ar_predictions, encoder_inputs = self.model(x)
    #
    #             all_y_hat.append(y_hat.squeeze(dim=2).cpu().numpy())
    #             all_y.append(y.cpu().numpy())
    #             all_ar_predictions.append(ar_predictions.squeeze(dim=2).cpu().numpy())
    #             all_residuals.append(residual.squeeze(dim=2).cpu().numpy())
    #             all_encoder_inputs.extend([ei.cpu().numpy() for ei in encoder_inputs])
    #
    #     # 反归一化
    #     y_hat_unnormalized = self.dataprepare.un_standardize(np.concatenate(all_y_hat))
    #     y_unnormalized = self.dataprepare.un_standardize(np.concatenate(all_y))
    #     ar_predictions_unnormalized = self.dataprepare.un_standardize(np.concatenate(all_ar_predictions))
    #     residuals_unnormalized = self.dataprepare.un_standardize(np.concatenate(all_residuals))
    #
    #     # 创建一个DataFrame
    #     df = pd.DataFrame({
    #         'Actual': y_unnormalized.flatten(),
    #         'Predicted': y_hat_unnormalized.flatten(),
    #         # 'AR Predictions': ar_predictions_unnormalized.flatten(),
    #         # 'Residuals': residuals_unnormalized.flatten()
    #     })
    #
    #     filename = f'prediction_results_{self.model_tag}.xlsx'
    #     df.to_excel(filename, index=False)
    #     print(f"预测结果保存至 {filename}")
    #
    #     return y_hat_unnormalized, y_unnormalized, ar_predictions_unnormalized, residuals_unnormalized, all_encoder_inputs
    #
    #

    def predict(self):
        self.model.eval()
        all_y_hat = []
        all_y = []
        all_ar_predictions = []
        all_residuals = []
        all_encoder_inputs = []

        # （可选）重置 CUDA 显存峰值计数  # NEW
        if DEVICE.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device=DEVICE)

        with torch.no_grad():
            for i, (x, y) in enumerate(self.test_generator):
                x, y = x.to(DEVICE), y.to(DEVICE)
                y_hat, residual, ar_predictions, encoder_inputs = self.model(x)

                all_y_hat.append(y_hat.squeeze(dim=2).cpu().numpy())
                all_y.append(y.cpu().numpy())
                all_ar_predictions.append(ar_predictions.squeeze(dim=2).cpu().numpy())
                all_residuals.append(residual.squeeze(dim=2).cpu().numpy())
                all_encoder_inputs.extend([ei.cpu().numpy() for ei in encoder_inputs])

        # ===== 在这里导出分阶段效率报告（一次即可） =====  # NEW
        import os, pandas as pd
        results_dir = os.path.join(os.getcwd(), "results")
        os.makedirs(results_dir, exist_ok=True)
        csv_path = os.path.join(results_dir, f"per_stage_efficiency_{self.model_tag}.csv")
        df = self.model.profiler.summarize(save_csv_path=csv_path)
        print("\n[Per-Stage Efficiency]")
        print(df.to_string(index=False))
        print(f"[Profiler] Report saved to: {csv_path}")
        # ===============================================

        # 反归一化（你原来的逻辑）
        y_hat_unnormalized = self.dataprepare.un_standardize(np.concatenate(all_y_hat))
        y_unnormalized = self.dataprepare.un_standardize(np.concatenate(all_y))
        ar_predictions_unnormalized = self.dataprepare.un_standardize(np.concatenate(all_ar_predictions))
        residuals_unnormalized = self.dataprepare.un_standardize(np.concatenate(all_residuals))

        df_out = pd.DataFrame({
            'Actual': y_unnormalized.flatten(),
            'Predicted': y_hat_unnormalized.flatten(),
        })
        filename = f'prediction_results_{self.model_tag}.xlsx'
        df_out.to_excel(filename, index=False)
        print(f"预测结果保存至 {filename}")

        return y_hat_unnormalized, y_unnormalized, ar_predictions_unnormalized, residuals_unnormalized, all_encoder_inputs
