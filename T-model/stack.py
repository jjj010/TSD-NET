import torch
import torch.nn as nn

from .basic_block import BasicBlock
from utils.profiler import StageProfiler


class Stack(nn.Module):
    """
    TSD-NET stack.

    Terminology used in the paper/rebuttal:
    "two-stage" = baseline trend estimation + residual refinement.
    Multiple DGR blocks are iterative refinement units inside the second stage,
    not additional third/fourth forecasting stages.
    """

    def __init__(
        self,
        input_size: int,
        encoder_channels,
        input_seqlen: int,
        forecast_seqlen: int,
        num_blocks: int = 3,
        kernel_size: int = 4,
        dropout: float = 0.2,
        enable_profiler: bool = True,
    ):
        super().__init__()
        if num_blocks < 1:
            raise ValueError("num_blocks must be >= 1")
        self.num_blocks = int(num_blocks)
        self.forecast_seqlen = int(forecast_seqlen)
        self.input_seqlen = int(input_seqlen)

        self.block_first = BasicBlock(
            input_size=input_size,
            encoder_num_channels=encoder_channels,
            forecast_seqlen=forecast_seqlen,
            estimate_seqlen=input_seqlen,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.blocks_rest = nn.ModuleList([
            BasicBlock(
                input_size=1,
                encoder_num_channels=encoder_channels,
                forecast_seqlen=forecast_seqlen,
                estimate_seqlen=input_seqlen,
                kernel_size=kernel_size,
                dropout=dropout,
            )
            for _ in range(num_blocks - 1)
        ])

        self.fusion_score_mlp = nn.Sequential(
            nn.Linear(2 * forecast_seqlen, forecast_seqlen),
            nn.ReLU(),
            nn.Linear(forecast_seqlen, forecast_seqlen),
        )

        self.profiler = StageProfiler() if enable_profiler else None
        if self.profiler is not None:
            self._register_profiler_hooks()

    def _register_profiler_hooks(self) -> None:
        def register_block(block: BasicBlock, idx: int):
            prefix = f"block{idx}"
            self.profiler.register_stage(block.encoder, f"{prefix}/ENCODER")
            self.profiler.add_params_of(block.encoder, f"{prefix}/ENCODER")
            for dec_name, dec in (("DECODER_F", block.f_decoder), ("DECODER_E", block.e_decoder)):
                self.profiler.register_stage(dec.rnn, f"{prefix}/{dec_name}")
                self.profiler.add_params_of(dec.rnn, f"{prefix}/{dec_name}")
                self.profiler.register_stage(dec.fc, f"{prefix}/{dec_name}")
                self.profiler.add_params_of(dec.fc, f"{prefix}/{dec_name}")
            self.profiler.register_stage(block.hidden_to_input, f"{prefix}/DECODER_E")
            self.profiler.add_params_of(block.hidden_to_input, f"{prefix}/DECODER_E")
            self.profiler.register_stage(block.ar_layer, f"{prefix}/AR_HEAD")
            self.profiler.add_params_of(block.ar_layer, f"{prefix}/AR_HEAD")
            self.profiler.register_stage(block.dag, f"{prefix}/DAG")
            self.profiler.add_params_of(block.dag, f"{prefix}/DAG")
            self.profiler.register_stage(block.gds_to_horizon, f"{prefix}/GDS_PROJ")
            self.profiler.add_params_of(block.gds_to_horizon, f"{prefix}/GDS_PROJ")

        register_block(self.block_first, 1)
        for i, b in enumerate(self.blocks_rest, start=2):
            register_block(b, i)
        self.profiler.register_stage(self.fusion_score_mlp, "GATED_FUSION")
        self.profiler.add_params_of(self.fusion_score_mlp, "GATED_FUSION")
        self.profiler.register_macs_hooks(self)


    def forward_ar_only(self, inputs: torch.Tensor) -> torch.Tensor:
        """Fast AR-only pass for Stage 1 pretraining.

        To keep all AR heads trainable without invoking the LSTM decoders, later
        blocks receive the observed load channel as their input proxy.
        """
        ar_predictions = [self.block_first.ar_forward(inputs)]
        current = inputs[:, :, 0:1]
        for block in self.blocks_rest:
            ar_predictions.append(block.ar_forward(current))
        return torch.stack(ar_predictions, dim=1).mean(dim=1)  # [B, H, 1]

    def forward(self, inputs: torch.Tensor):
        forecasts, gdss, ars, estimates, block_inputs = [], [], [], [], []
        forecast, residual, ar_pred, estimate, input_main, gds_h = self.block_first(inputs)
        forecasts.append(forecast); gdss.append(gds_h); ars.append(ar_pred)
        estimates.append(estimate); block_inputs.append(input_main)

        for block in self.blocks_rest:
            forecast, residual, ar_pred, estimate, input_main, gds_h = block(residual)
            forecasts.append(forecast); gdss.append(gds_h); ars.append(ar_pred)
            estimates.append(estimate); block_inputs.append(input_main)

        forecast_stack = torch.stack(forecasts, dim=1)  # [B, D, H, 1]
        gds_stack = torch.stack(gdss, dim=1)            # [B, D, H, 1]
        ar_stack = torch.stack(ars, dim=1)              # [B, D, H, 1]

        scores = []
        for m_j, gds_j in zip(forecasts, gdss):
            fusion_input = torch.cat([m_j.squeeze(-1), gds_j.squeeze(-1)], dim=-1)
            scores.append(self.fusion_score_mlp(fusion_input))
        score_stack = torch.stack(scores, dim=1)  # [B, D, H]
        fusion_weights = torch.softmax(score_stack, dim=1).unsqueeze(-1)
        final_forecast = torch.sum(fusion_weights * forecast_stack, dim=1)  # [B, H, 1]

        aux = {
            "forecast_stack": forecast_stack,
            "gds_stack": gds_stack,
            "ar_stack": ar_stack,
            "estimate_outputs": estimates,
            "block_inputs": block_inputs,
            "fusion_weights": fusion_weights,
        }
        ar_mean = ar_stack.mean(dim=1)
        return final_forecast, gds_stack, ar_mean, aux
