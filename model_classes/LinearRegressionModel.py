import torch
from torch import nn
import random

class LinearRegressionModel(nn.Module):
  def __init__(self, weights: float = random.random() * 2 - 1, bias: float = random.random() * 2 - 1):
    super().__init__()
    self.weights = nn.Parameter(torch.tensor([weights], requires_grad=True, dtype=torch.float))
    self.bias = nn.Parameter(torch.tensor([bias], requires_grad=True, dtype=torch.float))
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.weights * x + self.bias