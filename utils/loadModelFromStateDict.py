import torch
from torch import nn
from pathlib import Path

def loadModelFromStateDict(fileName: str, modelClass: nn.Module) -> nn.Module :
  FULL_STATE_DICT_PATH = Path('models/state_dict/' + fileName)
  model: nn.Module = modelClass()
  model.load_state_dict(torch.load(FULL_STATE_DICT_PATH))
  return model