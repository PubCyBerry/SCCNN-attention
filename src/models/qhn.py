import qiskit
from qiskit import transpile, assemble
import numpy as np
from torch.autograd import Function
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Dict

from src.layers import SCCNN, SCCNN_bn, Additive_Attention, Decoder

class QuantumCircuit:
    """
    This class provides a simple interface for interaction
    with the quantum circuit
    """

    def __init__(self, n_qubits, backend, shots, is_cnot=False):
        # --- Circuit definition ---
        self._circuit = qiskit.QuantumCircuit(n_qubits)

        all_qubits = [i for i in range(n_qubits)]
        # self.theta = qiskit.circuit.Parameter("theta")
        # --- multi qubit ---#
        self.theta = [qiskit.circuit.Parameter(f"theta_{i}") for i in all_qubits]

        self._circuit.h(all_qubits)
        self._circuit.barrier()
        if is_cnot:
            assert len(all_qubits) == 2, "CNOT must be with 2-qubits"
            self._circuit.cx(0, 1)
        # self._circuit.ry(self.theta, all_qubits)
        # --- multi qubit ---#
        for theta, qubit in zip(self.theta, all_qubits):
            self._circuit.ry(theta, qubit)

        self._circuit.measure_all()
        # ---------------------------

        self.backend = backend
        self.shots = shots

    def run(self, thetas):
        t_qc = transpile(self._circuit, self.backend)
        # qobj = assemble(
        #     t_qc,
        #     shots=self.shots,
        #     parameter_binds=[{self.theta: theta} for theta in thetas],
        # )
        # --- multi qubit -- #
        qobj = assemble(
            t_qc,
            shots=self.shots,
            parameter_binds=[{self.theta[i]: thetas[i] for i in range(len(thetas))}],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()

        counts = np.array(list(result.values()))
        # states = np.array(list(result.keys())).astype(float)
        # --- multi qubit -- #
        states = np.array([list(state) for state in result.keys()]).astype(float)

        # Compute probabilities for each state
        probabilities = counts / self.shots
        # Get state expectation
        # expectation = np.sum(states * probabilities)
        # -- multi qubit -- #
        expectation = np.sum(states * probabilities.reshape(-1, 1), axis=0)

        return np.array([expectation])

class HybridFunction(Function):
    """Hybrid quantum - classical function definition"""

    @staticmethod
    def forward(ctx, input, quantum_circuit, shift):
        """Forward pass computation"""
        ctx.shift = shift
        ctx.quantum_circuit = quantum_circuit

        # expectation_z = ctx.quantum_circuit.run(input[0].tolist())
        # result = torch.tensor([expectation_z])
        # -- multi qubit -- #
        expectation_z = [
            torch.Tensor(ctx.quantum_circuit.run(input[i].tolist()))
            for i in range(input.size(0))
        ]
        result = torch.concat(expectation_z, axis=0)
        ctx.save_for_backward(input, result)

        return result

    @staticmethod
    def backward(ctx, grad_output):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        """Backward pass computation"""
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
        gradients = torch.concat(gradients, axis=0)

        return (gradients * grad_output.float()).to(device), None, None


class Hybrid(nn.Module):
    """Hybrid quantum - classical layer definition"""

    def __init__(self, n_qubits, backend, shots, shift, is_cnot=False):
        super(Hybrid, self).__init__()
        self.quantum_circuit = QuantumCircuit(n_qubits, backend, shots, is_cnot=is_cnot)
        self.shift = shift

    def forward(self, input):
        return HybridFunction.apply(input, self.quantum_circuit, self.shift)



class QHN_roi_rank(nn.Module):
    def __init__(
        self, params: Optional[Dict] = None, shift=0.6, n_qubit=1, add_qubit=0
    ) -> None:
        super().__init__()

        self.roi_rank = params.roi_rank
        self.num_roi = len(params.roi_rank)
        self.lstm_hidden = params.LSTM.hidden_size
        self.n_qubit = n_qubit
        self.add_qubit = add_qubit

        self.hybrid = Hybrid(
            qiskit.Aer.get_backend("qasm_simulator"),
            shots=100,
            shift=shift,
            n_qubit=n_qubit,
            add_qubit=add_qubit,
        )
        # Hybrid class를 불러옴, Aer_simulator를 사용하고 shot은 100, shift는 pi/2(변화 가능), qubit의 수 지정

        params.LSTM.input_size = params.encoder.conv_block4.out_f
        params.decoder.input_dim = self.lstm_hidden * 2
        params.decoder.hidden_dim = n_qubit + add_qubit

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
        x, (h, c) = self.lstm(x)
        x = torch.concat(
            [x[:, -1, : self.lstm_hidden], x[:, 0, self.lstm_hidden :]], axis=1
        )
        x = x.flatten(1)
        # (batch, num_roi * last_channel) => (batch_size, 2)
        x = self.decoder(x)
        return x

        
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
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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
