import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalConv1d(nn.Module):
    """1D convolution with causal padding (pad on the left only)."""
    def __init__(self, in_channels, out_channels, kernel_size=2, dilation=1):
        super(CausalConv1d, self).__init__()
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                               dilation=dilation)
        # No padding argument here; we will pad in forward for causality.

    def forward(self, x):
        # Pad input on the left (past) side only.
        pad_amount = (self.kernel_size - 1) * self.dilation
        if pad_amount > 0:
            # Pad format is (left_pad, right_pad) for 1D conv.
            x = F.pad(x, (pad_amount, 0))
        return self.conv(x)

class WaveNetModel(nn.Module):
    def __init__(self, residual_channels=32, skip_channels=32, dilation_layers=10):
        """
        WaveNet-style model.
        residual_channels: number of channels in residual convolution branches.
        skip_channels: number of channels for skip connections.
        dilation_layers: how many dilated conv layers (with dilation doubling each layer).
        """
        super(WaveNetModel, self).__init__()
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.dilation_layers = dilation_layers

        # Initial 1x1 convolution to expand input from 1 channel to residual_channels.
        self.input_conv = nn.Conv1d(1, residual_channels, kernel_size=1)

        # Create lists for gated convolution layers
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs   = nn.ModuleList()

        for i in range(dilation_layers):
            # Dilation doubles each layer: 1, 2, 4, ...
            dilation = 2 ** i
            # Causal conv for filter and gate
            self.filter_convs.append(CausalConv1d(residual_channels, residual_channels,
                                                  kernel_size=2, dilation=dilation))
            self.gate_convs.append(CausalConv1d(residual_channels, residual_channels,
                                                kernel_size=2, dilation=dilation))
            # 1x1 conv for residual connection
            self.residual_convs.append(nn.Conv1d(residual_channels, residual_channels, kernel_size=1))
            # 1x1 conv for skip connection (to accumulate outputs for final output)
            self.skip_convs.append(nn.Conv1d(residual_channels, skip_channels, kernel_size=1))

        # Final output layers: combine skip outputs and map to 1-channel audio output
        self.output_mixer = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(skip_channels, 1, kernel_size=1)
        )

    def forward(self, x):
        # x shape: (batch, 1, samples)
        x = self.input_conv(x)  # shape: (batch, residual_channels, samples)
        skip_sum = 0  # accumulate skip connections output
        for i in range(self.dilation_layers):
            # Gated convolution
            filt = torch.tanh(self.filter_convs[i](x))
            gate = torch.sigmoid(self.gate_convs[i](x))
            out = filt * gate  # element-wise gating

            # Skip connection output
            skip_out = self.skip_convs[i](out)    # shape: (batch, skip_channels, samples)
            if isinstance(skip_sum, int) or skip_sum is None:
                skip_sum = skip_out
            else:
                skip_sum = skip_sum + skip_out    # accumulate skip outputs

            # Residual connection to feed into next layer
            res = self.residual_convs[i](out)     # (batch, residual_channels, samples)
            x = x + res                           # add to input (residual connection)

        # Mix skip outputs and produce final output
        y = self.output_mixer(skip_sum)  # shape: (batch, 1, samples)
        return y
