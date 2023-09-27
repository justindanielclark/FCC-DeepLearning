import torch
from torch import nn
from pathlib import Path

def loadModel(fileName: str, modelClass: nn.Module) -> nn.Module :
  FULL_MODEL_PATH = Path('models/full/' + fileName)
  return torch.load(f=FULL_MODEL_PATH)