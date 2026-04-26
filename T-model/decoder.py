import torch
import torch.nn as nn


class LSTMDecoder(nn.Module):
    """
    Single-layer LSTM decoder.

    The original paper describes the forecasting and estimation decoders as
    LSTM-based. This implementation keeps the naming and internal module
    consistent with that description.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int = 1):
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.output_size = int(output_size)
        self.rnn = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            bias=True,
            batch_first=True,
        )
        self.fc = nn.Linear(self.hidden_size, self.output_size, bias=True)

    def _to_lstm_state(self, prev_hidden):
        if isinstance(prev_hidden, tuple):
            return prev_hidden
        h0 = prev_hidden.contiguous()
        c0 = torch.zeros_like(h0)
        return h0, c0

    def forward(self, inputs: torch.Tensor, prev_hidden):
        """
        Args:
            inputs: [B, 1, input_size]
            prev_hidden: [1, B, hidden_size] or an LSTM state tuple.
        Returns:
            prediction: [B, 1, output_size]
            hidden: LSTM state tuple.
        """
        output, hidden = self.rnn(inputs, self._to_lstm_state(prev_hidden))
        prediction = self.fc(output.squeeze(dim=1)).unsqueeze(dim=1)
        return prediction, hidden
