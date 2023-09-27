import torch
from torch import nn

#! Literally the exact same model as LinearRegressionModel, just using built in torch.nn layers

class LinearRegressionModelV2(nn.Module):
  def __init__(self):
    super().__init__()
    # Creating the model parameters / also called: linear transform, probing layer, fully connected layer
    self.linear_layer = nn.Linear(in_features=1, out_features=1)
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.linear_layer(x)
