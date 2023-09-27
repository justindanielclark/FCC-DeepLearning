##
# ^ Exercise
# ? 6. Create a multi-class dataset using the spirals data creation function from CS231n
# ? 6.1 - Construct a model capable of fitting the data
# ? 6.2 - Build a loss function and optimizer capable of handling multiclass data
# ? 6.3 - Make a training and testing loop for the multi-class data and train a model on it to reach over 95% accuracy
# ? 6.4 - Plot the decision boundaries on the spiral dataset from your model predictions
import sys

sys.path.insert(0, "/home/jc/Desktop/Projects/Python/FCC-DeepLearning")
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn
from utils.printTensor import printTensor

device = "cuda" if torch.cuda.is_available() else "cpu"
N = 300  # number of points per class
D = 2  # dimensionality
K = 5  # number of classes
X = np.zeros((N * K, D))  # data matrix (each row = single example)
y = np.zeros(N * K, dtype="uint8")  # class labels
for j in range(K):
    ix = range(N * j, N * (j + 1))
    r = np.linspace(0.0, 1, N)  # radius
    t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2  # theta
    X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
    y[ix] = j
# Visualizing The Data
# plt.style.use("dark_background")
# plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
# plt.show()
# Shaping The Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train = torch.from_numpy(X_train).type(torch.float32).to(device)
X_test = torch.from_numpy(X_test).type(torch.float32).to(device)
y_train = torch.from_numpy(y_train).type(torch.LongTensor).to(device)
y_test = torch.from_numpy(y_test).type(torch.LongTensor).to(device)
# Model
model = nn.Sequential(
    nn.Linear(in_features=2, out_features=16),
    nn.ReLU(),
    nn.Linear(in_features=16, out_features=K),
)
model.to(device)
# Loss Function / Optimizer / Accuracy
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.15)
# Training Loop / Testing Loop
EPOCHS = 1000
for epoch in range(EPOCHS):
    model.train()
    y_logits = model(X_train)
    y_preds = torch.softmax(y_logits, dim=1).argmax(dim=1).type(torch.LongTensor).to(device)
    loss = loss_function(y_logits, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model.eval()
    with torch.inference_mode():
        y_test_logits = model(X_test)
        y_test_preds = torch.softmax(y_test_logits, dim=1).argmax(dim=1).type(torch.LongTensor)
        test_loss = loss_function(y_test_logits, y_test)
        if epoch % 10 == 0:
          print(f"Epoch: {epoch} | Loss: {loss:.2f} | Test_Loss: {test_loss:.2f}")

## Plotted
from utils.helperFunctions import plot_decision_boundary
plt.style.use("dark_background")
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title("Train")
plot_decision_boundary(model, X_train, y_train)
plt.subplot(1,2,1)
plt.title("Test")
plot_decision_boundary(model, X_test, y_test)
plt.show()