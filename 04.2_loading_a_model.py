from pathlib import Path
import torch
from torch import nn

class LinearRegressionModel(nn.Module):
  def __init__(self):
    super().__init__()

    self.weights = nn.Parameter(torch.randn(1,
                                            requires_grad=True,
                                            dtype=torch.float))
    self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.weights * x + self.bias

## NEEDS ACCESS TO THE MODEL ITS LOADING EITHER DECLARED OUR THRU IMPORT
FULL_MODEL_PATH = Path('models/full/04_pytorch_workflow_model.pt')
model: LinearRegressionModel = torch.load(f=FULL_MODEL_PATH)