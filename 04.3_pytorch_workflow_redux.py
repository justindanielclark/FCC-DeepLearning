import torch
from torch import nn
from utils.printTensor import printTensor as pp

device = "cuda" if torch.cuda.is_available() else "cpu"
#^ Generating Test Data
X = torch.arange(0, 10, .2).unsqueeze(dim=1).to(device)
WEIGHT = .5
BIAS = .3
y = torch.add(torch.mul(X, torch.tensor(WEIGHT)), torch.tensor(BIAS))
training_split = int(len(X) * .8)
X_train = X[:training_split]
X_test = X[training_split:]
y_train = y[:training_split]
y_test = y[training_split:]
pp(X_train, "X_train")
pp(X_test, "X_test")
pp(y_train, "y_train")
pp(y_test, "y_test")

#^ Create Model, Loss Function, Optimizer
class LinearModel(nn.Module):
  def __init__(self) -> None:
    super().__init__()
    self.linear_layer = nn.Linear(in_features=1, out_features=1)
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.linear_layer(x).to(device)

#? Set Model to Device
model = LinearModel()
model.to(device)
lossFunction = nn.L1Loss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)

#^ Start Training
EPOCHS = 300
for epoch in range(EPOCHS):
  model.train()
  y_pred = model(X_train)
  loss = lossFunction(y_pred, y_train)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  model.eval()
  with torch.inference_mode():
    test_pred = model(X_test)
    test_loss = lossFunction(test_pred, y_test)
    if(epoch % 10 == 0):
      print(f"Epoch: {epoch} | Loss: {loss} | Test Loss: {test_loss}")

#^ Start Training With A Finer Learning Rate
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001)
EPOCHS = 300
for epoch in range(EPOCHS):
  model.train()
  y_pred = model(X_train)
  loss = lossFunction(y_pred, y_train)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  model.eval()
  with torch.inference_mode():
    test_pred = model(X_test)
    test_loss = lossFunction(test_pred, y_test)
    if(epoch % 10 == 0):
      print(f"Epoch: {epoch} | Loss: {loss} | Test Loss: {test_loss}")

#^ Start Training With An Even Finer Learning Rate
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.0001)
EPOCHS = 300
for epoch in range(EPOCHS):
  model.train()
  y_pred = model(X_train)
  loss = lossFunction(y_pred, y_train)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  model.eval()
  with torch.inference_mode():
    test_pred = model(X_test)
    test_loss = lossFunction(test_pred, y_test)
    if(epoch % 10 == 0):
      print(f"Epoch: {epoch} | Loss: {loss} | Test Loss: {test_loss}")

#^ Start Training With An Even Finer Learning Rate
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.00001)
EPOCHS = 300
for epoch in range(EPOCHS):
  model.train()
  y_pred = model(X_train)
  loss = lossFunction(y_pred, y_train)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  model.eval()
  with torch.inference_mode():
    test_pred = model(X_test)
    test_loss = lossFunction(test_pred, y_test)
    if(epoch % 10 == 0):
      print(f"Epoch: {epoch} | Loss: {loss} | Test Loss: {test_loss}")

#^ Start Training With An Even Finer Learning Rate
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.000001)
EPOCHS = 300
for epoch in range(EPOCHS):
  model.train()
  y_pred = model(X_train)
  loss = lossFunction(y_pred, y_train)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  model.eval()
  with torch.inference_mode():
    test_pred = model(X_test)
    test_loss = lossFunction(test_pred, y_test)
    if(epoch % 10 == 0):
      print(f"Epoch: {epoch} | Loss: {loss} | Test Loss: {test_loss}")
