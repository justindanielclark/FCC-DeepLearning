import torch
from torch import nn
import random

class QuadradicModel(nn.Module):
  def __init__(self, weight: float = (random.random() * 2) - 1 , bias: float = (random.random() * 2) - 1):
    super().__init__()
    self.weight = nn.Parameter(torch.tensor([weight], dtype=torch.float, requires_grad=True)) 
    self.bias = nn.Parameter(torch.tensor([bias], dtype=torch.float, requires_grad=True))
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.weight * torch.mul(x, x) + self.bias