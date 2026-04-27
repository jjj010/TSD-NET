import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
    def __init__(self, input_size: int, horizon: int, hidden_size: int = 64):
        super().__init__()
        self.horizon = int(horizon)
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, self.horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.rnn(x)
        y = self.fc(h[-1])
        return y.unsqueeze(-1)


class CNNForecaster(nn.Module):
    def __init__(self, input_size: int, horizon: int, hidden_size: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(hidden_size, int(horizon))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x.permute(0, 2, 1)).squeeze(-1)
        return self.fc(z).unsqueeze(-1)


class TransformerForecaster(nn.Module):
    def __init__(self, input_size: int, horizon: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.proj = nn.Linear(input_size, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, int(horizon))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(self.proj(x))
        pooled = z[:, -1, :]
        return self.fc(pooled).unsqueeze(-1)
