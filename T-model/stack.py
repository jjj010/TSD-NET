import torch
import torch.nn as nn
from .basic_block import BasicBlock
# stack.py 顶部
from utils.profiler import StageProfiler


class Stack(nn.Module):
    def __init__(self,
                 input_size: int,
                 encoder_channels,
                 input_seqlen: int,
                 forecast_seqlen: int,
                 num_blocks: int = 3):
        """
        :param input_size: 第一个 block 的输入维度
        :param encoder_channels: 每个 block 里 encoder 的通道数列表
        :param input_seqlen: 输入序列长度
        :param forecast_seqlen: 输出预测序列长度
        :param num_blocks: block 的总数 (>=1)
        """
        super(Stack, self).__init__()

        self.num_blocks = num_blocks
        self.forecast_seqlen = forecast_seqlen

        # 第一个block的输入维度是input_size
        self.block_first = BasicBlock(
            input_size=input_size,
            encoder_num_channels=encoder_channels,
            forecast_seqlen=forecast_seqlen,
            estimate_seqlen=input_seqlen
        )

        # 后续block的输入维度都固定为1，可根据需要修改
        self.blocks_rest = nn.ModuleList()
        for _ in range(num_blocks - 1):
            block = BasicBlock(
                input_size=1,
                encoder_num_channels=encoder_channels,
                forecast_seqlen=forecast_seqlen,
                estimate_seqlen=input_seqlen
            )
            self.blocks_rest.append(block)

        # gated fusion MLP 仅基于 m_j 本身
        self.fusion_weight_mlp = nn.Sequential(
            nn.Linear(forecast_seqlen, forecast_seqlen),
            nn.ReLU(),
            nn.Linear(forecast_seqlen, forecast_seqlen),
            nn.Sigmoid()
        )
        self.profiler = StageProfiler()

        def _register_block_stages(block, idx):
            # ENCODER
            if hasattr(block, "encoder"):
                self.profiler.register_stage(block.encoder, f"block{idx}/ENCODER")
                self.profiler.add_params_of(block.encoder, f"block{idx}/ENCODER")

            # ----- DECODER-F (forecast) -----
            if hasattr(block, "f_decoder"):
                if hasattr(block.f_decoder, "gru"):
                    self.profiler.register_stage(block.f_decoder.gru, f"block{idx}/DECODER_F")
                    self.profiler.add_params_of(block.f_decoder.gru, f"block{idx}/DECODER_F")
                if hasattr(block.f_decoder, "fc"):
                    # fc 也算到 DECODER（避免和 AR 重复）
                    self.profiler.register_stage(block.f_decoder.fc, f"block{idx}/DECODER_F")
                    self.profiler.add_params_of(block.f_decoder.fc, f"block{idx}/DECODER_F")

            # ----- DECODER-E (estimate) -----
            if hasattr(block, "e_decoder"):
                if hasattr(block.e_decoder, "gru"):
                    self.profiler.register_stage(block.e_decoder.gru, f"block{idx}/DECODER_E")
                    self.profiler.add_params_of(block.e_decoder.gru, f"block{idx}/DECODER_E")
                if hasattr(block.e_decoder, "fc"):
                    self.profiler.register_stage(block.e_decoder.fc, f"block{idx}/DECODER_E")
                    self.profiler.add_params_of(block.e_decoder.fc, f"block{idx}/DECODER_E")

            # hidden_to_input 也归到 DECODER（属于解码相关的预处理）
            if hasattr(block, "hidden_to_input"):
                self.profiler.register_stage(block.hidden_to_input, f"block{idx}/DECODER_E")
                self.profiler.add_params_of(block.hidden_to_input, f"block{idx}/DECODER_E")

            # ----- AR 头（你专门的自回归层） -----
            if hasattr(block, "ar_layer"):
                self.profiler.register_stage(block.ar_layer, f"block{idx}/AR_HEAD")
                self.profiler.add_params_of(block.ar_layer, f"block{idx}/AR_HEAD")

            # ----- DAG 门控 -----
            if hasattr(block, "dag"):
                self.profiler.register_stage(block.dag, f"block{idx}/DAG")
                self.profiler.add_params_of(block.dag, f"block{idx}/DAG")

        # 对第一个和后续 block 逐个登记  # NEW

        _register_block_stages(self.block_first, 1)
        for i, b in enumerate(self.blocks_rest, start=2):
            _register_block_stages(b, i)

        # 让 Linear/Conv1d/GRU 具备 MACs 统计钩子  # NEW
        self.profiler.register_macs_hooks(self)
    def forward(self, inputs):
        """
        融合所有 block 的预测结果，使用 gated fusion 权重计算：
        \hat{Y} = \sum_j \alpha_j \cdot m_j，其中 \alpha_j = \text{MLP}(m_j)
        """
        forecasts = []

        # 第一个 block
        forecast, residual, ar_predictions, estimate_outputs, _ = self.block_first(inputs)
        forecasts.append(forecast)

        # 后续 blocks
        for block in self.blocks_rest:
            forecast_i, residual, _, _, _ = block(residual)
            forecasts.append(forecast_i)

        # Gated fusion，仅基于 m_j
        final_forecast = 0
        for m_j in forecasts:
            alpha_j = self.fusion_weight_mlp(m_j.squeeze(-1)).unsqueeze(-1)  # [B, T, 1]
            final_forecast += alpha_j * m_j

        return final_forecast, residual, ar_predictions, estimate_outputs


