import numpy as np
import pandas as pd
from torch import nn
import torch
from torchviz import make_dot


class BasicLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(input_dim, output_dim)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        return x


class TicTacPlayer(nn.Module):
    def __init__(self, *params):
        super().__init__()

        self.model = nn.Sequential()
        for i in range(len(params) - 1):
            self.model.append(BasicLayer(params[i], params[i + 1]))

    def forward(self, x):
        x = self.model(x)
        return x


field = np.array([[0, 1, -1], [1, 0, -1], [-1, 1, 0]])
model = TicTacPlayer(9, 7, 5, 7, 10)
out = model(torch.Tensor(field.flatten()))

make_dot(out, params=dict(model.named_parameters())).render("model_structure", format="svg", view=True)
