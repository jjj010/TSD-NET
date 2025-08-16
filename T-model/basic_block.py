import torch
import torch.nn as nn
from .encoder import TcnEncoder
from .decoder import GruDecoder

class BasicBlock(nn.Module):
    def __init__(self, input_size, encoder_num_channels, forecast_seqlen, estimate_seqlen, kernel_size=5):
        super(BasicBlock, self).__init__()
        self.input_size = input_size
        self.encoder_num_channels = encoder_num_channels
        self.hidden_size = encoder_num_channels[-1]
        self.kernel_size = kernel_size

        self.forecast_seqlen = forecast_seqlen
        self.estimate_seqlen = estimate_seqlen

        # 编码器
        self.encoder = TcnEncoder(num_inputs=self.input_size, num_channels=self.encoder_num_channels,
                                  kernel_size=self.kernel_size)

        # 预测解码器
        self.f_decoder = LSTMDecoder(input_size=1, hidden_size=self.hidden_size, output_size=1)
        # 估计解码器
        self.e_decoder = LSTMDecoder(input_size=1, hidden_size=self.hidden_size, output_size=1)

    
        self.ar_layer = nn.Linear(self.hidden_size, self.forecast_seqlen)

      
        self.hidden_to_input = nn.Linear(self.hidden_size, 1)


        self.dag = nn.Sequential(
            nn.Linear(estimate_seqlen, estimate_seqlen),
            nn.ReLU(),
            nn.Linear(estimate_seqlen, estimate_seqlen),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        
        batch_size = inputs.shape[0]

        hidden = self.encoder(inputs)
        hidden = hidden.contiguous()

     
        ar_predictions = self.ar_layer(hidden.squeeze(0))  # [B, forecast_seqlen]
        ar_predictions = ar_predictions.unsqueeze(2)  # [B, forecast_seqlen, 1]


        f_decoder_input = ar_predictions[:, 0, :].unsqueeze(1)
        e_decoder_input = hidden.permute(1, 0, 2)  # [B, 1, H]
        e_decoder_input = self.hidden_to_input(e_decoder_input)  # [B, 1, 1]

        forecast_outputs = []
        estimate_outputs = []

        f_hidden = hidden
        for _ in range(self.forecast_seqlen):
            out, f_hidden = self.f_decoder(f_decoder_input, f_hidden)
            forecast_outputs.append(out)
            f_decoder_input = out

        e_hidden = hidden
        for _ in range(self.estimate_seqlen):
            out, e_hidden = self.e_decoder(e_decoder_input, e_hidden)
            estimate_outputs.append(out)
            e_decoder_input = out

        forecast_outputs = torch.cat(forecast_outputs, dim=2)  # [B, 1, T]
        estimate_outputs = torch.cat(estimate_outputs, dim=2)  # [B, 1, T]


        input_main = inputs[:, 0, :].unsqueeze(1)  # [B, 1, T]

        input_main = input_main.permute(0, 2, 1)                 # [B, T, 1]
        estimate_outputs = estimate_outputs.permute(0, 2, 1)     # [B, T, 1]

        
        delta = input_main - estimate_outputs                   # [B, T, 1]
        delta_flat = delta.squeeze(-1)                          # [B, T]
        alpha_dag = self.dag(delta_flat)                        # [B, T]
        gds = alpha_dag.unsqueeze(-1) * delta                   # [B, T, 1]

        return forecast_outputs, input_main - gds, ar_predictions, estimate_outputs, input_main