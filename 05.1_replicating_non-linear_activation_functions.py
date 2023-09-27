import torch
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

t0 = torch.arange(-10,10, 1, dtype=torch.float32)

def relu(x: torch.Tensor) -> torch.Tensor:
  return torch.maximum(torch.tensor(0), x)

def sigmoid(x: torch.Tensor) -> torch.Tensor:
  return torch.divide(torch.tensor(1), torch.add(torch.tensor(1), torch.exp(-x))) 

t1 = relu(t0)
t2 = sigmoid(t0)

plt.title("Standard")
plt.plot(t0)
plt.show()

plt.title("ReLU")
plt.plot(t1)
plt.show()

plt.title("Sigmoid")
plt.plot(t2)
plt.show()