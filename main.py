import torch
import torch.nn as nn
import os
from prepare_data.prepare_data import DataPrepare
from model.stack import Stack
from utils.train_utils import Train
from utils.metrics import MyMetrics


# 定义超参数
HyperParams = {
    'datapath': './prepare_data/data',
    'datafile': 'shh',
    'split_ratio': [0.8, 0.1, 0.1],
    "batch_size": 24,
    "N_EPOCHS": 1,
    'lr': 5e-3,
    "features": 2,
    "input_seqlen": 24,
    "predict_seqlen": 24
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float64)

def main():
    load_model = False

    # 初始化模型
    model = Stack(
        input_size=HyperParams['features'],
        encoder_channels=[4, 6, 8, 12],
        input_seqlen=HyperParams['input_seqlen'],
        forecast_seqlen=HyperParams['predict_seqlen']
    ).to(DEVICE)


    optimizer = torch.optim.Adam(model.parameters(), lr=HyperParams['lr'])
    loss_func = nn.MSELoss(reduction='mean')


    T = Train(
        hyperparams=HyperParams,
        model=model,
        optimizer=optimizer,
        loss_func=loss_func,
        data_prepare_cls=DataPrepare,  # 传入你的数据准备类
        model_tag='block3F2'
    )


    if load_model:
        T.load_state()


    T.train_model()


    y_hat, y, ar_predictions, residual, encoder_inputs = T.predict()


    mymetrics = MyMetrics(y_hat, y, T.dataprepare)
    mymetrics.print_metrics()




if __name__ == "__main__":
    main()
