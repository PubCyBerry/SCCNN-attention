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


from torch.autograd import Function
import qiskit
from qiskit import transpile, assemble
from qiskit.visualization import *
from qiskit import IBMQ
from qiskit.providers.aer import AerSimulator

mode = 0


class QuantumCircuit:
    """
    This class provides a simple interface for interaction
    with the quantum circuit
    """

    def __init__(self, n_qubits, backend, shots):
        # --- Circuit definition ---
        self._circuit = qiskit.QuantumCircuit(n_qubits)

        all_qubits = [i for i in range(n_qubits)]
        self.theta = qiskit.circuit.Parameter("theta")

        self._circuit.h(all_qubits)
        self._circuit.barrier()
        self._circuit.ry(self.theta, all_qubits)

        self._circuit.measure_all()
        # ---------------------------

        self.backend = backend
        self.shots = shots

    def run(self, batch_thetas):
        if mode == 1:
            if len(np.shape(batch_thetas)) == 1:
                batch_thetas = [batch_thetas.tolist()]
            else:
                batch_thetas = batch_thetas.tolist()

            batchExpectations = list()
            for thetas in batch_thetas:
                t_qc = transpile(self._circuit, self.backend)
                qobj = assemble(
                    t_qc,
                    shots=self.shots,
                    parameter_binds=[{self.theta: theta} for theta in thetas],
                )
                job = self.backend.run(qobj)
                result = job.result().get_counts()

                counts = np.array(list(result.values()))
                states = np.array(list(result.keys())).astype(float)

                # Compute probabilities for each state
                probabilities = counts / self.shots
                # Get state expectation
                expectation = np.sum(states * probabilities)
                batchExpectations.append(expectation)

            return np.array(batchExpectations)

        else:
            t_qc = transpile(self._circuit, self.backend)
            qobj = assemble(
                t_qc,
                shots=self.shots,
                parameter_binds=[{self.theta: theta} for theta in batch_thetas],
            )
            job = self.backend.run(qobj)
            result = job.result().get_counts()

            counts = np.array(list(result.values()))
            states = np.array(list(result.keys())).astype(float)

            # Compute probabilities for each state
            probabilities = counts / self.shots
            # Get state expectation
            expectation = np.sum(states * probabilities)

            return np.array([expectation])


class HybridFunction(Function):
    """Hybrid quantum - classical function definition"""

    @staticmethod
    def forward(ctx, input, quantum_circuit, shift):
        """Forward pass computation"""
        ctx.shift = shift
        ctx.quantum_circuit = quantum_circuit

        if mode == 1:
            expectation_z = ctx.quantum_circuit.run(input)
            result = torch.tensor(expectation_z)
            if input.size() != result.size():
                result = result.view(input.size())
            ctx.save_for_backward(input, result)
        else:
            expectation_z = ctx.quantum_circuit.run(input[0].tolist())
            result = torch.tensor([expectation_z])
            ctx.save_for_backward(input, result)

        return result

    @staticmethod
    def backward(ctx, grad_output):
        if mode == 1:
            """Backward pass computation"""
            input, expectation_z = ctx.saved_tensors
            input_list = torch.ones_like(input)

            shift_right = input_list + input_list * ctx.shift
            shift_left = input_list - input_list * ctx.shift
            gradients = []
            for i in range(len(input_list)):
                expectation_right = ctx.quantum_circuit.run(shift_right[i])
                expectation_left = ctx.quantum_circuit.run(shift_left[i])

                gradient = torch.tensor(expectation_right) - torch.tensor(
                    expectation_left
                )
                gradients.append(gradient)
            gradients = torch.Tensor(np.array(gradients, dtype=np.float32)).view_as(
                input
            )
            # gradients = gradients.to(device)
            # grad_output = grad_output.to(device)
            return gradients * grad_output, None, None

        else:
            input, expectation_z = ctx.saved_tensors
            input_list = np.array(input.tolist())

            shift_right = input_list + np.ones(input_list.shape) * ctx.shift
            shift_left = input_list - np.ones(input_list.shape) * ctx.shift

            gradients = []
            for i in range(len(input_list)):
                expectation_right = ctx.quantum_circuit.run(shift_right[i])
                expectation_left = ctx.quantum_circuit.run(shift_left[i])

                gradient = torch.tensor([expectation_right]) - torch.tensor(
                    [expectation_left]
                )
                gradients.append(gradient)
            gradients = np.array([gradients]).T
            return torch.tensor([gradients]).float() * grad_output.float(), None, None


class Hybrid(nn.Module):
    """Hybrid quantum - classical layer definition"""

    def __init__(self, backend, shots, shift):
        super(Hybrid, self).__init__()
        self.quantum_circuit = QuantumCircuit(1, backend, shots)
        self.shift = shift

    def forward(self, input):
        return HybridFunction.apply(input, self.quantum_circuit, self.shift)


class QNN(nn.Module):
    def __init__(self, *args: Any, **kwargs: Any):
        super(QNN, self).__init__()
        simulator = qiskit.Aer.get_backend("aer_simulator")
        # simulator.set_options(device="GPU")
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 1)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gapfc1 = nn.Linear(16, 64)
        self.gapfc2 = nn.Linear(64, 1)
        self.hybrid = Hybrid(simulator, 100, np.pi / 2)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        # x = x.flatten(1)
        # x = F.relu(self.fc1(x))
        # x = self.fc2(x)
        x = self.gap(x)
        x = x.flatten(1)
        x = F.relu(self.gapfc1(x))
        x = self.gapfc2(x)
        x = self.hybrid(x)
        return torch.cat((x, 1 - x), -1)


class SCCNN_attention_QNN(nn.Module):
    def __init__(self, params: Optional[Dict] = None) -> None:
        super().__init__()
        simulator = qiskit.Aer.get_backend("aer_simulator")

        self.num_roi = 116
        params.decoder.input_dim = self.num_roi * params.encoder.conv_block4.out_f

        self.uniEncoder = params.is_encoder_shared
        if self.uniEncoder:
            self.encoder = SCCNN_bn(params.encoder)
        else:
            self.encoder = nn.ModuleList(
                [SCCNN_bn(params.encoder) for i in range(self.num_roi)]
            )

        self.attention = Additive_Attention(**params.attention)
        params.decoder.output_dim = 1
        self.decoder = Decoder(**params.decoder)
        self.hybrid = Hybrid(simulator, 100, np.pi / 2)

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
        x = self.hybrid(x)
        return torch.cat((x, 1 - x), -1)


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

    from src.data import SITES_DICT, ROIDataset
    from src.datamodules import LOSODataModule
    from copy import deepcopy

    qnn = QNN()
    conf = OmegaConf.load("/workspace/Configs/config.yaml")
    conf.merge_with_cli()
    for i, (key, value) in enumerate(SITES_DICT.items()):
        train_site = deepcopy(list(SITES_DICT.keys()))
        test_site = train_site.pop(i)
        conf.data.train_site = train_site
        conf.data.test_site = test_site

        dm = LOSODataModule(conf.data, conf.loader, ROIDataset)
        dm.setup()
        for j, (x, y) in enumerate(dm.val_dataloader()):
            # print(
            #     "{}, {}, {}, {}".format(
            #         x.size(), x.is_pinned(), y.size(), y.is_pinned()
            #     )
            # )
            # for i, bins in enumerate(np.bincount(y)):
            #     print(f"{i}: {bins:2d} ", end="")
            # print()
            out = qnn(x)
            print('SITE',i, 'iter',j, out, y, out.size(), y.size())
