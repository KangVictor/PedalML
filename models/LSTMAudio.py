import torch
import torch.nn as nn

# Define your LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)


    def forward(self, x, hidden=None):
        """
        Hidden state is for passing in the previous hidden state.
        Audio Size is too large to be parsed at once.
        """
        if hidden is None:
            # Initialize zero vector for initial hidden state
            batch_size = x.size(0)
            h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
            c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
            hidden = (h_0, c_0)

        out, hidden = self.lstm(x, hidden)  # out: (B, T, H)
        out = self.fc(out)                  # out: (B, T, 1)
        return out, hidden
