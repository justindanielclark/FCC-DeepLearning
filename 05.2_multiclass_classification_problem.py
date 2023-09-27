import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.datasets import (
    make_blobs,
)  # ? https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html
from sklearn.model_selection import train_test_split
from utils.helperFunctions import accuracy_fn
from utils.printTensor import printTensor
from utils.helperFunctions import plot_decision_boundary

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Create Multi-Class Data
N_SAMPLES = 4000
NUM_CLASSES = 10
NUM_FEATURES = 2
RANDOM_SEED = 420
CLUSTER_NOISE = .75
STANDARD_HIDDEN_LAYER_FEATURES = 126
LEARNING_RATE = 0.1
EPOCHS = 2000

X, y = make_blobs(
    n_samples=N_SAMPLES,
    n_features=NUM_FEATURES,
    centers=NUM_CLASSES,
    random_state=RANDOM_SEED,
    cluster_std=CLUSTER_NOISE,
)
# 2 Create Tensors / Test and Training Data
X = torch.from_numpy(X).type(torch.float32)
y = torch.from_numpy(y).type(torch.LongTensor)

# 3. Split into Train and Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED
)

# 4. Plot Data
plt.style.use("dark_background")
plt.figure(figsize=(10,7))
plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.RdYlBu)
plt.show()

# 5. Create Model
model = nn.Sequential(
    nn.Linear(in_features=NUM_FEATURES, out_features=STANDARD_HIDDEN_LAYER_FEATURES),
    nn.ELU(),
    nn.Linear(
        in_features=STANDARD_HIDDEN_LAYER_FEATURES,
        out_features=STANDARD_HIDDEN_LAYER_FEATURES,
    ),
    nn.Softplus(),
    nn.Linear(
        in_features=STANDARD_HIDDEN_LAYER_FEATURES,
        out_features=STANDARD_HIDDEN_LAYER_FEATURES,
    ),
    nn.Linear(in_features=STANDARD_HIDDEN_LAYER_FEATURES, out_features=NUM_CLASSES),
)

# 6. Put Everything On The Same Target Device (GPU if possible)
model.to(device)
X_train = X_train.to(device)
X_test = X_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)

# 7. Pick a Loss Function and Optimizer
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=LEARNING_RATE)

# 8. Getting Prediction Probabilities For A Multi-Class PyTorch Model
model.eval()
with torch.inference_mode():
  y_pred_prob = torch.argmax(torch.softmax(model(X_train[:25]), 1), dim=1)
  printTensor(y_pred_prob, 'y_pred_prob')

# 9. Training and Testing Loops

# for lr in range(10):
#   optimizer = torch.optim.SGD(params=model.parameters(), lr=LEARNING_RATE/(lr+1))
for epoch in range(EPOCHS):
    model.train()
    y_logits = model(X_train)
    y_preds = torch.softmax(y_logits, dim=1).argmax(dim=1)
    loss = loss_function(y_logits, y_train)
    acc = accuracy_fn(y_true=y_train, y_pred=y_preds)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model.eval()
    with torch.inference_mode():
        y_test_logits = model(X_test)
        y_test_preds = torch.softmax(y_test_logits, dim=1).argmax(dim=1)
        test_loss = loss_function(y_test_logits, y_test)
        test_acc = accuracy_fn(y_true=y_test, y_pred=y_test_preds)
        if epoch % 10 == 0:
            print(f"Epoch: {epoch} | Loss: {loss:.5f} | Acc: {acc:.2f}% | Test_Loss: {test_loss:.5f} | Test_Acc: {test_acc:.5f}%")

model.train()
y_logits = model(X_test[:5])
printTensor(y_logits, "y_logits")
y_preds = torch.softmax(y_logits, dim=1)
printTensor(y_preds, "y_preds")
printTensor(y_test[:5], "y_test")


# 10. Visualize
plt.style.use("dark_background")
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title("Train")
plot_decision_boundary(model, X_train, y_train)
plt.subplot(1,2,1)
plt.title("Test")
plot_decision_boundary(model, X_test, y_test)
plt.show()