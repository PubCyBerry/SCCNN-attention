from re import U
from typing import Optional, Dict, Any
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.layers import SCCNN, SCCNN_bn, Additive_Attention, Decoder


class SCCNN_attention(nn.Module):
    def __init__(self, params: Optional[Dict] = None) -> None:
        super().__init__()

        self.num_roi = len(params.roi_rank)
        params.decoder.input_dim = self.num_roi * params.encoder.conv_block4.out_f

        self.uniEncoder = params.is_encoder_shared
        if self.uniEncoder:
            self.encoder = SCCNN_bn(params.encoder)
        else:
            self.encoder = nn.ModuleList(
                [SCCNN_bn(params.encoder) for i in range(self.num_roi)]
            )

        self.attention = Additive_Attention(**params.attention)
        self.decoder = Decoder(**params.decoder)

    def forward(self, x) -> torch.Tensor:
        # (batch, num_roi, time_series) => (batch, last_channel, num_roi)
        if self.uniEncoder:
            x = torch.concat(
                [self.encoder(x[:, roi, :]) for roi in range(self.num_roi)], axis=2
            )
        else:
            x = torch.concat(
                [self.encoder[roi](x[:, roi, :]) for roi in range(self.num_roi)], axis=2
            )
        


        # (batch, last_channel, num_roi) => (batch, num_roi * last_channel)
        x = self.attention(x).flatten(1)
        # (batch, num_roi * last_channel) => (batch_size, 2)
        x = self.decoder(x)
        return x

class SCCNN_LSTM(nn.Module):
    def __init__(self, params: Optional[Dict] = None) -> None:
        super().__init__()

        self.num_roi = len(params.roi_rank)
        self.lstm_hidden = params.LSTM.hidden_size
        params.LSTM.input_size = params.encoder.conv_block4.out_f
        params.decoder.input_dim = self.lstm_hidden * 2

        self.uniEncoder = params.is_encoder_shared
        if self.uniEncoder:
            self.encoder = SCCNN_bn(params.encoder)
        else:
            self.encoder = nn.ModuleList(
                [SCCNN_bn(params.encoder) for i in range(self.num_roi)]
            )

        self.lstm = nn.LSTM(**params.LSTM)
        self.decoder = Decoder(**params.decoder)

    def forward(self, x) -> torch.Tensor:
        # (batch, num_roi, time_series) => (batch, num_roi, last_channel)
        if self.uniEncoder:
            x = [self.encoder(x[:, roi, :]) for roi in range(self.num_roi)]
        else:
            x = [self.encoder[roi](x[:, roi, :]) for roi in range(self.num_roi)]
        
        if len(x) == 1:
            x = x[0]
        else:
            x = torch.concat(x, axis=2)
        
        x = x.permute(0, 2, 1)
        # (batch, num_roi, last_channel) => (batch, lstm_hidden * 2)
        x, (h,c) = self.lstm(x)
        x = torch.concat([x[:, -1, :self.lstm_hidden], x[:,0, self.lstm_hidden:]], axis=1)
        x = x.flatten(1)
        # (batch, num_roi * last_channel) => (batch_size, 2)
        x = self.decoder(x)
        return x

        
class SCCNN_LSTM_roi_rank(nn.Module):
    def __init__(self, params: Optional[Dict] = None) -> None:
        super().__init__()

        self.roi_rank = params.roi_rank
        self.num_roi = len(params.roi_rank)
        self.lstm_hidden = params.LSTM.hidden_size
        params.LSTM.input_size = params.encoder.conv_block4.out_f
        params.decoder.input_dim = self.lstm_hidden * 2

        self.uniEncoder = params.is_encoder_shared
        if self.uniEncoder:
            self.encoder = SCCNN_bn(params.encoder)
        else:
            self.encoder = nn.ModuleList(
                [SCCNN_bn(params.encoder) for i in range(self.num_roi)]
            )

        self.lstm = nn.LSTM(**params.LSTM)
        self.decoder = Decoder(**params.decoder)

    def forward(self, x) -> torch.Tensor:
        # (batch, num_roi, time_series) => (batch, num_roi, last_channel)
        if self.uniEncoder:
            x = [self.encoder(x[:, roi, :]) for roi in self.roi_rank]
        else:
            x = [self.encoder[roi](x[:, roi, :]) for roi in self.roi_rank]
        
        if len(x) == 1:
            x = x[0]
        else:
            x = torch.concat(x, axis=2)
        
        x = x.permute(0, 2, 1)
        # (batch, num_roi, last_channel) => (batch, lstm_hidden * 2)
        x, (h,c) = self.lstm(x)
        x = torch.concat([x[:, -1, :self.lstm_hidden], x[:,0, self.lstm_hidden:]], axis=1)
        x = x.flatten(1)
        # (batch, num_roi * last_channel) => (batch_size, 2)
        x = self.decoder(x)
        return x


if __name__ == "__main__":
    import os
    from omegaconf import OmegaConf

    config = OmegaConf.load(os.path.join("Configs", "config.yaml"))
    model_params = OmegaConf.load(os.path.join("Configs", "models.yaml"))
    params = OmegaConf.merge(config, model_params)
    model = SCCNN_attention(params.network)

    # print(model)
    print(model(torch.randn((32, 116, 176))).size())
    print(model(torch.randn((32, 116, 236))).size())