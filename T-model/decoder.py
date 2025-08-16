
import torch
import torch.nn as nn


# 解码器
class LSTMDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        """
        解码器
        :param input_size: 在解码器部分输入序列的维度 为 1
        :param hidden_size: 需同上下文向量特征维度相同 ，即编码器参数 num_channels[-1]
        :param output_size: 结果线性层的输出维度  为1
        """
        super(LSTMDecoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.gru = nn.LSTM(input_size=self.input_size,
                          hidden_size=self.hidden_size,
                          num_layers=1,
                          bias=True,
                          batch_first=True)
        self.fc = nn.Linear(in_features=self.hidden_size, out_features=self.output_size, bias=True)

    def forward(self, inputs, prev_hidden):
        """
        params:
            inputs: [batch_size, features=1, seq_len=1]
            prev_hidden [1, batch_size, hidden_size]
        returns:
            prediction: [batch_size, feature=1， seq_len=1]
            hidden: [1, batch_size, hidden_size]
        """
        # output [batch_size, features=hidden_size, seq_len=1]
        # hidden [1, batch_size, hidden_size]
        output, hidden = self.gru(inputs, prev_hidden)
        # prediction [batch_size, output_size=1, seq_len=1]
        prediction = self.fc(output.squeeze(dim=1)).unsqueeze(dim=1)
        return prediction, hidden