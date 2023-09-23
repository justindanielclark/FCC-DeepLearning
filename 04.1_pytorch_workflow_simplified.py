
import random
import torch
from torch import nn
from utils.plotPredictions import plot_predictions as pp

#^ Data (Preparing and Loading)
X = torch.arange(-10, 10, .02).unsqueeze(dim = 1)
weight = random.random() * 2 - 1
bias = random.random() * 2 - 1
y = weight * torch.mul(X, X) + bias
train_split = int(.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

#^ Build Model
class QuadradicModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.weights = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float)) 
    self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.weights * torch.mul(x, x) + self.bias
  
model = QuadradicModel()

with torch.no_grad():
  y_pred = model(X_test)

#^ Loss Functions and Optimizers
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001)

#^ Print Intial Dataset
pp(X_train, y_train, X_test, y_test, y_pred)

epochLimit = 1000
curr = 0
while curr < epochLimit:
  #^ Train
  model.train()
  y_pred = model(X_train)
  loss = loss_fn(y_pred, y_train)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  #^ Test
  model.eval()
  with torch.inference_mode():
    test_pred = model(X_test)
    test_loss = loss_fn(test_pred, y_test)
  curr += 1

#^ Print Resultant Dataset With Test Predictions
pp(X_train, y_train, X_test, y_test, test_pred)