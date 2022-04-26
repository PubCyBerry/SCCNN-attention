import qiskit
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Dict

from src.layers import SCCNN, SCCNN_bn
from src.layers import Additive_Attention
from src.layers import Hybrid_Decoder, Hybrid


class Simple_QHN(nn.Module):
    def __init__(self, params: Optional[Dict] = None, *args, **kwargs) -> None:
        super(Simple_QHN, self).__init__()
        params = params.Simple_QHN
        self.lstm_hidden = params.lstm_hidden
        self.conv1 = nn.Conv1d(116, 6, kernel_size=5)
        self.conv2 = nn.Conv1d(6, 16, kernel_size=5)
        self.dropout = nn.Dropout2d()
        self.lstm = nn.LSTM(
            input_size=16,
            hidden_size=self.lstm_hidden,
            num_layers=1,
            bidirectional=True,
        )
        self.fc1 = nn.Linear(256, params.linear_out)
        self.fc2 = nn.Linear(params.linear_out, params.n_qubits)
        self.hybrid = Hybrid(
            params.n_qubits,
            qiskit.Aer.get_backend("aer_simulator"),
            100,
            shift=params.shift,
            is_cnot=params.is_cnot,
        )
        self.fc3 = nn.Linear(params.n_qubits * 2, 2)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, 2)
        x = torch.flatten(x, 2)
        x, (h, c) = self.lstm(x.permute(0, 2, 1))
        x = torch.concat(
            [x[:, -1, : self.lstm_hidden], x[:, 0, self.lstm_hidden :]], axis=1
        )
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.tanh(x) * torch.ones_like(x) * torch.tensor(np.pi / 2)
        x = self.hybrid(x).to(self.device)
        x = torch.cat((x, 1 - x), -1)
        x = F.softmax(self.fc3(x), dim=1)
        return x


class SCCNN_LSTM_Hybrid(nn.Module):
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
        self.decoder = Hybrid_Decoder(**params.decoder, **params.hybrid)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        x, (h, c) = self.lstm(x)
        x = torch.concat(
            [x[:, -1, : self.lstm_hidden], x[:, 0, self.lstm_hidden :]], axis=1
        )
        x = x.flatten(1)

        # hybrid
        # type 1: Densenet -> Gate q_0=control, q_1=target
        # type 2: Densenet + Gate q_0=control, q_1=target
        # (batch, num_roi * last_channel) => (batch_size, 2)
        x = self.decoder(x)
        return x


if __name__ == "__main__":
    import numpy as np

    n_qubit = 1
    add_qubit = 4
    shift = 0.6
    for batch_size in [1, 16, 32]:
        data = np.random.normal(0, 10, (batch_size, 5))
        data = F.tanh(torch.Tensor(data))
        H = Hybrid(
            qiskit.Aer.get_backend("qasm_simulator"),
            shots=100,
            shift=shift,
            n_qubit=n_qubit,
            add_qubit=add_qubit,
        )
        x = torch.stack([H(data[i : i + 1]).squeeze() for i in range(len(data))], 0)
        print(x.size(), data[0:1].size(), H(data[0:1]).size())
        # x = H(data)
        x = torch.stack((torch.ones_like(x) - x, x), 1).cuda()
        print(f"batch_size={batch_size}, size={x.size()}")
        y = torch.LongTensor([1, 2, 3, 4, 5, 6]).view(1, -1)
        # print(binary(y, bits=4))
