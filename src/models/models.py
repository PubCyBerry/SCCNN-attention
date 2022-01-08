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

    # def init_weights(self, m):
    #     if isinstance(m, nn.Conv1d):
    #         torch.nn.init.xavier_normal_(m.weight)


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