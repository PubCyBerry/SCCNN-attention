from typing import Any, Optional, List
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_block( in_f: int = 1, out_f: int = 1, activation: Optional[str] = None, dropout_p: float = 0, pool: int = 1, *args: Any, **kwargs: Any,) -> nn.Sequential:
    activations = nn.ModuleDict([["lrelu", nn.LeakyReLU()], ["relu", nn.ReLU()]])

    seq = OrderedDict()
    seq.update({"Conv": nn.Conv1d(in_f, out_f, *args, **kwargs)})
    if dropout_p > 0:
        seq.update({"Dropout": nn.Dropout(dropout_p)})
    if pool > 1:
        seq.update({"Pool": nn.MaxPool1d(pool)})
    if activation:
        seq.update({activation: activations[activation]})
    return nn.Sequential(seq)


class SCCNN(nn.Module):
    """
    Seperated Channel Convolution Neural Network
    """
    def __init__(
        self,
        block_params = {'conv_block1': {'in_f':  1, 'out_f': 32, 'kernel_size': 3, 'activation': 'lrelu', 'dropout_p': 0.3, 'pool': 2},
                        'conv_block2': {'in_f': 32, 'out_f': 64, 'kernel_size': 3, 'activation': 'lrelu', 'dropout_p': 0.3, 'pool': 2},
                        'conv_block3': {'in_f': 64, 'out_f': 96, 'kernel_size': 3, 'activation': 'lrelu', 'dropout_p': 0.0, 'pool': 0},
                        'conv_block4': {'in_f': 96, 'out_f': 96, 'kernel_size': 3, 'activation': 'lrelu', 'dropout_p': 0.3, 'pool': 0}},
        padding: str = "valid",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super(SCCNN, self).__init__()
        self.conv_block1 = conv_block(**block_params['conv_block1'], padding=padding)
        self.conv_block2 = conv_block(**block_params['conv_block2'], padding=padding)
        self.conv_block3 = conv_block(**block_params['conv_block3'], padding=padding)
        self.conv_block4 = conv_block(**block_params['conv_block4'], padding=padding)
        self.GAP = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        """
        x: (batch, time_series)
        """
        # (batch, 1, time_series)
        x = x.unsqueeze(1)
        # (batch, 32, time_series-2 / 2)
        x = self.conv_block1(x)
        # (batch, 64, prev-2 / 2)
        x = self.conv_block2(x)
        # (batch, 96, prev-2)
        x = self.conv_block3(x)
        # (batch, 96, prev-2)
        x = self.conv_block4(x)
        # (batch, 96, 1)
        x = self.GAP(x)
        return x

        
class Additive_Attention(nn.Module):
    def __init__(self, input_dim: int = 96, attention_dim: int = 256, output_dim: int = 1, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.Wg1 = nn.Linear(input_dim, attention_dim, bias=False)
        self.Wg2 = nn.Linear(input_dim, attention_dim, bias=False)
        self.bg  = nn.Parameter(torch.zeros(attention_dim))

        self.Wa  = nn.Linear(attention_dim, output_dim)

    def forward(self, x):
        # (batch, channel, 116) => (batch, 116, channel)
        x = torch.transpose(x, 1, 2)
        # (batch, 116, 1, channel) + (batch,  1, 116, channel) => (batch, 116, 116, channel)
        gnn = torch.tanh(self.Wg1(x).unsqueeze(2) + self.Wg2(x).unsqueeze(1) + self.bg)
        # (batch, 116, 116)
        att = torch.sigmoid(self.Wa(gnn)).squeeze()

        # (batch, 116, channel)
        return torch.matmul(att, x)

    
class Decoder(nn.Module):
    def __init__(self, input_dim: int=512, hidden_dim: int=512, output_dim: int=2, dropout_p: float=0, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.decoder = nn.Sequential(
            OrderedDict(
                [
                    ['linear', nn.Linear(input_dim, hidden_dim)],
                    ['relu', nn.ReLU()],
                    ['dropout', nn.Dropout(dropout_p)],
                    ['linear_out', nn.Linear(hidden_dim, output_dim)],
                    ['log_softmax', nn.LogSoftmax(dim=1)]
                ]
            )
        )

    def forward(self, x):
        x = self.decoder(x)
        return x


if __name__ == "__main__":
    import torch

    model = SCCNN()
    print(model)
    print(model(torch.randn((32, 176))).size())
    print(model(torch.randn((32, 236))).size())