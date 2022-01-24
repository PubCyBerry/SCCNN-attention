from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.layers import SCCNN, Additive_Attention, Decoder


class SCCNN_attention(nn.Module):
    def __init__(self, params: Optional[Dict]=None) -> None:
        super().__init__()

        self.num_roi = 116
        params.decoder.input_dim = self.num_roi * params.encoder.conv_block4.out_f
        
        self.encoder = nn.ModuleList([SCCNN(params.encoder) for i in range(self.num_roi)])
        self.attention = Additive_Attention(**params.attention)
        self.decoder = Decoder(**params.decoder)

    def forward(self, x) -> torch.Tensor:
        # (batch, num_roi, time_series) => (batch, last_channel, num_roi)
        x = torch.concat([self.encoder[roi](x[:,roi,:]) for roi in range(self.num_roi)], axis=2)
        # (batch, last_channel, num_roi) => (batch, num_roi * last_channel)
        x = self.attention(x).flatten(1)
        # (batch, num_roi * last_channel) => (batch_size, 2)
        x = self.decoder(x)
        return x


class SCCNN_attention(nn.Module):
    def __init__(self, params: Optional[Dict]=None) -> None:
        super().__init__()

        self.num_roi = 116
        params.decoder.input_dim = self.num_roi * params.encoder.conv_block4.out_f
        
        self.encoder = nn.ModuleList([SCCNN(params.encoder) for i in range(self.num_roi)])
        self.attention = Additive_Attention(**params.attention)
        self.decoder = Decoder(**params.decoder)

    def forward(self, x) -> torch.Tensor:
        # (batch, num_roi, time_series) => (batch, last_channel, num_roi)
        x = torch.concat([self.encoder[roi](x[:,roi,:]) for roi in range(self.num_roi)], axis=2)
        # (batch, last_channel, num_roi) => (batch, num_roi * last_channel)
        x = self.attention(x).flatten(1)
        # (batch, num_roi * last_channel) => (batch_size, 2)
        x = self.decoder(x)
        return x

    # def init_weights(self, m):
    #     if isinstance(m, nn.Conv1d):
    #         torch.nn.init.xavier_normal_(m.weight)

    
from typing import Any
class MNISTNet(nn.Module):
    def __init__(self, *args: Any, **kwargs: Any):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


if __name__ == '__main__':
    import os
    from omegaconf import OmegaConf

    config = OmegaConf.load(os.path.join('config','config.yaml'))
    model_params = OmegaConf.load(os.path.join('config','models.yaml'))
    params = OmegaConf.merge(config, model_params)
    model = SCCNN_attention(params.network)

    # print(model)
    print(model(torch.randn((32, 116, 176))).size())
    print(model(torch.randn((32, 116, 236))).size())