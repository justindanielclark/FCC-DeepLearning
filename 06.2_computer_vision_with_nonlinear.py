import torch
from torch import nn
from utils.trainingAndTesting import (test_step, train_step)
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import ToTensor
from utils.helperFunctions import accuracy_fn
import tqdm
from timeit import default_timer as timer

#^ Get Data
train_data = torchvision.datasets.FashionMNIST(
    root="data",  # ? Where to Save This
    train=True,  # ? Do We Want The Training Set
    download=True,  # do we want to download?
    transform=ToTensor(),  # How do we want to transform the data?
    target_transform=None,  # How do we want to transform the labels/targets
)
test_data = torchvision.datasets.FashionMNIST(
    root="data", train=False, download=True, transform=ToTensor(), target_transform=None
)
class_labels = train_data.classes
BATCH_SIZE = 32
train_dataloader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)

#^ Eval Model
def eval_Model(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module, accuracy_fn):
    """Returns a dictionary containing the results of model predicting on data_loader"""
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            # Make Predictions
            y_pred = model(X)
            # Accumulate the loss and acc values per batch
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_true = y, y_pred = y_pred.argMax(dim=1))
        # Scale loss and acc to find average loss/acc per batch
        loss /= len(data_loader)
        acc /= len(data_loader)
    return {
        "Model_Name": model.__class__.__name__,
        "Model_Loss": loss.item(),
        "Model_Acc": acc}

#^ Building a new model
device = "cuda" if torch.cuda.is_available() else "cpu"
class FashionMNISTModelV1(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.Linear(in_features=hidden_units, out_features=output_shape)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

modelV1 = FashionMNISTModelV1(input_shape=784, hidden_units=10, output_shape=10).to(device)

#^ Loss Function and Optimizer
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=modelV1.parameters(), lr=.1)

#^ Training Loop / Testing Loop

epochs = 3
time_start = timer()
for epoch in range(epochs):
    train_step(modelV1, train_dataloader, loss_function, optimizer, accuracy_fn, device)
    test_step(modelV1, test_dataloader, loss_function, optimizer, accuracy_fn, device)
time_end = timer()
print(f"Total Time To Run {epochs} Epochs: {(time_end-time_start):.2f} seconds")