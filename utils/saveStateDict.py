import torch
from torch import nn
from pathlib import Path

def saveStateDict(model: nn.Module, filename: str):
  MODELS_PATH = Path("models")
  MODELS_PATH.mkdir(parents=True, exist_ok=True)
  STATE_DICT_MODELS_PATH = Path("models/state_dict")
  STATE_DICT_MODELS_PATH.mkdir(parents=True, exist_ok=True)
  STATE_DICT_SAVE_PATH = STATE_DICT_MODELS_PATH / filename
  torch.save(obj=model.state_dict(), f=STATE_DICT_SAVE_PATH)