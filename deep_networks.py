from torch import nn, Tensor
from typing import Iterable

import torch


class Tanh2(nn.Module):

    def forward(self, input: Tensor) -> Tensor:
        return 2 * torch.tanh(input)

class FCNetwork(nn.Module):

    def __init__(self, dims: Iterable[int], output_activation: nn.Module = None):

        super().__init__()
        self.input_size = dims[0]
        self.out_size = dims[-1]
        self.layers = self.make_seq(dims, output_activation)

    @staticmethod
    def make_seq(dims: Iterable[int], output_activation: nn.Module) -> nn.Module:
        mods = []

        for i in range(len(dims) - 2):
            mods.append(nn.Linear(dims[i], dims[i + 1]))
            mods.append(nn.ReLU())

        mods.append(nn.Linear(dims[-2], dims[-1]))
        if output_activation:
            mods.append(output_activation())
        return nn.Sequential(*mods)

    def forward(self, x: Tensor) -> Tensor:
        # Feedforward
        return self.layers(x)

    def hard_update(self, source: nn.Module):
        for target_param, source_param in zip(self.parameters(), source.parameters()):
            target_param.data.copy_(source_param.data)

    def soft_update(self, source: nn.Module, tau: float):
        for target_param, source_param in zip(self.parameters(), source.parameters()):
            target_param.data.copy_(
                (1 - tau) * target_param.data + tau * source_param.data
            )

