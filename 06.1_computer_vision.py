import torch
from torch import nn
from utils.printTensor import printTensor

# ^ Computer Vision Libraries in PyTorch
# ? torchvision - base domain library for PyTorch computer vision
# ? torchvision.datasets - get datasets and data loading functions for computer vision here
# ? torchvision.models - get pretrained computer vision models that you can leverage for your own problems
# ? torchvision.transforms - functinons for manipulating your vision data (images) to be suitable for use with an ML model
# ? torch.utils.data.Dataset - base dataset class for Pytorch
# ? torch.utils.data.DataLoader - creates a Python iterable over a dataset

import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor

# ^ Getting a Dataset
# ? FasionMNIST
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
print(class_labels)

# ? Visualize Our Data
import matplotlib.pyplot as plt

# fig = plt.figure(figsize=(9,9))
# rows, cols = 4,4
# for i in range(1, rows * cols + 1):
#   random_idx = torch.randint(0, len(train_data), size=[1]).item()
#   img, label = train_data[random_idx]
#   fig.add_subplot(rows, cols, i)
#   plt.title(class_labels[label])
#   plt.imshow(img.squeeze(), cmap="gray")
#   plt.axis(False)
# plt.show()

# ^ Prepare Data Loader
# ? A DataLoader turns our dataset into a Python iteratble
# ? More Specifically, we want to turn our data into batches (or-minibatches)
# ? Why would we do this?
# ? Our current dataset is 60k images. To hold all 60 in memory is expensive.
# ? It gives our neural network more chances to update its gradients per epoch (you only get one backwards SGD update per epoch)

from torch.utils.data import DataLoader

BATCH_SIZE = 32
train_dataloader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)

print(f"DataLoaders: {train_dataloader, test_dataloader}")
print(f"Length of train_dataloader: {len(train_dataloader)} batches of {BATCH_SIZE}...")
print(f"Length of test_dataloader: {len(test_dataloader)} batches of {BATCH_SIZE}...")

train_features_batch, train_labels_batch = next(iter(train_dataloader))
print(f"train_features_batch.shape: {train_features_batch.shape}")
print(f"train_labels_batch.shape: {train_labels_batch.shape}")

# Show a Sample
random_idx = torch.randint(0, len(train_features_batch), size=[1]).item()
img, label = train_features_batch[random_idx], train_labels_batch[random_idx]

plt.title(class_labels[label])
plt.imshow(img.squeeze(), cmap="gray")
plt.show()

# ^ Build a Baseline Model
# When starting to build a series of machine learning modeeling experiements, its best practice to start with a baseline model
# A baseline model is a simple model you will try to improve upon with subseuqent models/experiments
# In other words, start simply and then add complexity

# ? We need to create a flatten layer first!
flatten_model = nn.Flatten()
# Lets get a single sample
x = train_features_batch[0]
printTensor(x, "sample train features tensor")
output = flatten_model(x).squeeze()
printTensor(output, "output after flattening...")  # 784 Input features


class FashionMNISTModelV0(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.Linear(in_features=hidden_units, out_features=output_shape),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_stack(x)


modelV0 = FashionMNISTModelV0(
    input_shape=784,  ## Size of Flattened Image
    hidden_units=10,  ## Num of Hidden Units
    output_shape=len(class_labels),  ## One for Every Class
)

# ^ Loss Function and Optimizer
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=modelV0.parameters(), lr=0.1)
from utils.helperFunctions import accuracy_fn

# ^ Creating a function to time our experiments
# ? Machine learning is experimental.
# ? Two of the main things you'll often want to track are:
# ? Model Performance
# ? How Fast It Runs

from timeit import default_timer as timer


def print_train_time(start: float, end: float, device: torch.device = None) -> float:
    """Prints difference between start and end time."""
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time


start_time = timer()
end_time = timer()
print_train_time(start=start_time, end=end_time, device="cpu")

#^ Creating a Training Loop and Training Our Model On Batches of Data
#? Loop Through Epochs
#? Loop through training batches, perform training steps, calculate train loss *per batch*
#? Loop through testing batches, perform testing steps, calculate the test loss *per batch*
#? Print out whats happening
#? Time it all (for fun)

from tqdm.auto import tqdm

#Set the seed and start the timer
torch.manual_seed(42)
train_time_start_on_cpu = timer()

#Set the number of epochs (we will kill this small for faster training time)
epochs = 3

# Create training and test loop
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n-----")
    ### Training
    train_loss = 0
    # Add a loop to loop through training batches
    for batch, (X,y) in enumerate(train_dataloader):
        modelV0.train()
        ## Forward Pass
        y_pred = modelV0(X)
        ## Calculate The loss 
        loss = loss_function(y_pred, y)
        train_loss += loss
        ## Optimizer Zero Grad
        optimizer.zero_grad()
        ## Loss Backward
        loss.backward()
        ## Optimizer Step
        optimizer.step()
        #Print out what's happening:
        if batch % 400 == 0:
            print(f"Looked at {batch * len(X)}/{len(train_dataloader.dataset)} samples.")
    train_loss /= len(train_dataloader)

    ### Testing
    test_loss, test_acc = 0, 0
    modelV0.eval()
    with torch.inference_mode():
        for X_test, y_test in test_dataloader:
            test_pred = modelV0(X_test)
            test_loss += loss_function(test_pred, y_test)
            test_acc += accuracy_fn(y_true=y_test, y_pred=test_pred.argmax(dim=1))
        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)
    print(f"\nTrain loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

train_time_end_on_cpu = timer()
total_train_time_modelV0 = print_train_time(start=train_time_start_on_cpu, end=train_time_end_on_cpu, device=str(next(modelV0.parameters()).device))


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

modelV0_results = eval_Model(model=modelV0, data_loader=test_dataloader, loss_fn=loss_function, accuracy_fn=accuracy_fn)

