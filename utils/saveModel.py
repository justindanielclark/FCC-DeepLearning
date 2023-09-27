import torch
from torch import nn
from pathlib import Path

def saveModel(model: nn.Module, filename: str):
  MODELS_PATH = Path("models")
  MODELS_PATH.mkdir(parents=True, exist_ok=True)
  FULL_MODELS_PATH = Path("models/full")
  FULL_MODELS_PATH.mkdir(parents=True, exist_ok=True)
  FULL_MODEL_SAVE_PATH = FULL_MODELS_PATH / filename
  torch.save(obj=model, f=FULL_MODEL_SAVE_PATH)