import torch
import torch.nn as nn
from .encoder import TcnEncoder
from .decoder import LSTMDecoder

class BasicBlock(nn.Module):
    """One Difference-Guided Refinement (DGR) block."""

    def __init__(
        self,
        input_size: int,
        encoder_num_channels,
        forecast_seqlen: int,
        estimate_seqlen: int,
        kernel_size: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_size = int(input_size)
        self.encoder_num_channels = list(encoder_num_channels)
        self.hidden_size = self.encoder_num_channels[-1]
        self.forecast_seqlen = int(forecast_seqlen)
        self.estimate_seqlen = int(estimate_seqlen)

        self.encoder = TcnEncoder(
            num_inputs=self.input_size,
            num_channels=self.encoder_num_channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.f_decoder = LSTMDecoder(input_size=1, hidden_size=self.hidden_size, output_size=1)
        self.e_decoder = LSTMDecoder(input_size=1, hidden_size=self.hidden_size, output_size=1)
        self.ar_layer = nn.Linear(self.hidden_size, self.forecast_seqlen)
        self.hidden_to_input = nn.Linear(self.hidden_size, 1)
        self.dag = nn.Sequential(
            nn.Linear(self.estimate_seqlen, self.estimate_seqlen),
            nn.ReLU(),
            nn.Linear(self.estimate_seqlen, self.estimate_seqlen),
            nn.Sigmoid(),
        )
        self.gds_to_horizon = nn.Linear(self.estimate_seqlen, self.forecast_seqlen)


    def ar_forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Fast AR-only forward used in Stage 1 without running decoders."""
        if inputs.ndim != 3:
            raise ValueError(f"Expected [B, T, F], got {tuple(inputs.shape)}")
        tcn_inputs = inputs.permute(0, 2, 1).contiguous()
        hidden = self.encoder(tcn_inputs)
        return self.ar_layer(hidden.squeeze(0)).unsqueeze(-1)  # [B, H, 1]

    def forward(self, inputs: torch.Tensor):
        """
        Args:
            inputs: [B, T, F] for first block; [B, T, 1] for later blocks.
        Returns:
            forecast_outputs: [B, H, 1]
            next_input: [B, T, 1]
            ar_predictions: [B, H, 1]
            estimate_outputs: [B, T, 1]
            input_main: [B, T, 1]
            gds_horizon: [B, H, 1]
        """
        if inputs.ndim != 3:
            raise ValueError(f"Expected [B, T, F], got {tuple(inputs.shape)}")

        tcn_inputs = inputs.permute(0, 2, 1).contiguous()
        hidden = self.encoder(tcn_inputs)  # [1, B, hidden]
        ar_predictions = self.ar_layer(hidden.squeeze(0)).unsqueeze(-1)  # [B, H, 1]

        f_decoder_input = ar_predictions[:, 0:1, :]
        forecast_outputs = []
        f_hidden = hidden
        for _ in range(self.forecast_seqlen):
            out, f_hidden = self.f_decoder(f_decoder_input, f_hidden)
            forecast_outputs.append(out)
            f_decoder_input = out
        forecast_outputs = torch.cat(forecast_outputs, dim=1)  # [B, H, 1]

        e_decoder_input = self.hidden_to_input(hidden.permute(1, 0, 2))
        estimate_outputs = []
        e_hidden = hidden
        for _ in range(self.estimate_seqlen):
            out, e_hidden = self.e_decoder(e_decoder_input, e_hidden)
            estimate_outputs.append(out)
            e_decoder_input = out
        estimate_outputs = torch.cat(estimate_outputs, dim=1)  # [B, T, 1]

        input_main = inputs[:, :, 0:1]
        dfb = input_main - estimate_outputs
        alpha_dag = self.dag(dfb.squeeze(-1)).unsqueeze(-1)
        gds_observed = alpha_dag * dfb
        next_input = input_main - gds_observed
        gds_horizon = self.gds_to_horizon(gds_observed.squeeze(-1)).unsqueeze(-1)

        return forecast_outputs, next_input, ar_predictions, estimate_outputs, input_main, gds_horizon
