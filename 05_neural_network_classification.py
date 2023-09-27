
import torch
from utils.printTensor import printTensor as pp
from utils.helperFunctions import plot_decision_boundary

#^ Neural Network Classification 
#? Three Types
  #? Binary Classification - Spam vs Not Spam
  #? Multiclass Classification - Is it Pizza, Steak, or Sushi?
  #? Multilabel Classification - What sort of tags should this article have?

#^ What will be covered
#? Architecture of a neural network classification model
#? Input shapes and output shapes of a classification model (features and labels)
#? Creating custom data to view, fit on, and predict on
#? Steps in modeling
  #? Creating a model, a loss function and optimizer, creating a training loop, evaluating a model
#? Saving and Loading models
#? Harnessing the power of non-linearity
#? Different classification models

#^ Classification inputs and ouputs (SEE 8:44:00)
#? Remember: Inputs -> Some Machine Learning Algorithm -> Output
#? In the case of images:
#?  Inputs (3Color Channels x 224W x 224H) -> Some Machine Learning Algo -> [0.97, 0.00, .03] (shape [3], for pizza sushi steak)

#^ Architecture of a classification model
#? See Image in helpful_images

#^ Make classification data and get it ready
from sklearn.datasets import make_circles

device = "cuda" if torch.cuda.is_available() else "cpu"

#^ Make 1000 samples of binary classification
n_samples = 1000
X, y = make_circles(n_samples, noise=0.03, random_state=42)

print("First 5 samples of X\n", X[:5])
print("First 5 samples of y\n", y[:5])

import pandas as pd
circles = pd.DataFrame({"X[0]": X[:, 0], "X[1]": X[:, 1], "label": y})
print(circles.head(5))

#^ Visualize
import matplotlib.pyplot as plt
# plt.scatter(x=X[:, 0], y=X[:,1], c=y, cmap=plt.cm.RdYlBu)
# plt.show()

import torch
X = torch.from_numpy(X).type(torch.float32).to(device)
y = torch.from_numpy(y).type(torch.float32).to(device)

#^ Random Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, train_size=.8, random_state=42)

print("\n", len(X_train), len(X_test), len(y_train), len(y_test), "\n")

pp(X_train, "X_train")
pp(X_test, "X_test")
pp(y_train, "y_train")
pp(y_test, "y_test")

#^ Make Model
#? We want to create a model that:
#? 1. Subclasses nn.Module
#? 2. Creates 2 `nn.Linear()` layers that are capable of handling the shapes of our data
#? 3. Defines a `forward()` method that outlines the forward pass (or forward computation)
#? 4. Intantiate a copy of the model and send it to the device
from torch import nn
class CirclePredictionModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.layers = [
      nn.Linear(in_features=2, out_features=5), 
      nn.Linear(in_features=5, out_features=1)
    ]
  def forward(self, x):
    return self.layers[1](self.layers[0](x))


model_0 = CirclePredictionModel().to(device)

#? Lets replicate the model using nn.Sequential
model_1 = nn.Sequential(
  nn.Linear(in_features=2, out_features=256),
  nn.ReLU(),
  nn.Linear(in_features=256, out_features=256),
  nn.ReLU(),
  nn.Linear(in_features=256, out_features=256),
  nn.ReLU(),
  nn.Linear(in_features=256, out_features=1),

).to(device)

#^ Setup Loss Function and Optimizer
#? Which loss function or optimizer should you use?
#? Again... this is problem specific
#? For Regression you miht want MAE or MSE (Mean absolute error or Mean Squared Error)
#? For Classification, you might want binary cross entropy or categorial cross entropy (Cross entropy)
#? As a reminder, the loss function measures how *wrong* your models predictions are
#? For Optimizers, two of the most common and useful are SGD and Adam
loss_function = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params=model_1.parameters(), lr = 0.1)

def accuracy_fn(y_true, y_pred):
  correct = torch.eq(y_true, y_pred).sum().item()
  acc = (correct/len(y_pred)) * 100
  return acc

#^ Getting the data in the right shape for testing/training
#? We have a problem with data types not matching
model_1.eval()
with torch.inference_mode():
  y_logits = model_1(X_test.to(device))[:5]
  pp(y_logits, 'y_logits')

  y_pred_probs = torch.sigmoid(y_logits)
  pp(y_pred_probs, 'y_pred_probs')

  y_preds = torch.round(y_pred_probs)
  pp(y_preds, 'y_preds')


EPOCHS = 1000
for epoch in range(EPOCHS):
  ##Training
  model_1.train()
  ##Forward Pass
  y_logits = model_1(X_train).squeeze()
  y_pred = torch.round(torch.sigmoid(y_logits))
  ## Calculate loss/accuracy
  # loss = loss_function(torch.sigmoid(y_logits), ## nn.BCELoss expects prediction probabilities
  #                      y_train)
  loss = loss_function(y_logits, # nn.BCEWithLogitsLoss expects raw logits as input
                       y_train)
  acc = accuracy_fn(y_true=y_train, y_pred=y_pred)
  ## Optimizer Zero Grad  
  optimizer.zero_grad()
  ## Loss Backward
  loss.backward()
  ## Optimizer Step(gradient descent)
  optimizer.step()
  model_1.eval()
  with torch.inference_mode():
    ##Forward Pass
    test_logits = model_1(X_test).squeeze()
    test_pred = torch.round(torch.sigmoid(test_logits))
    ## Calculate Test Loss / Accuracy
    test_loss = loss_function(test_logits, y_test)
    test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)
  ## Print out what's happening
  if epoch % 10 == 0:
    print(f"Epoch: {epoch} | Loss: {loss:.5f} | Acc: {acc:.2f}% | Test_Loss: {test_loss:.5f} | Test_Acc: {test_acc:.5f}")

#^ Our Model Is Not Learning Anything :(
#? Lets inspect it and make some predictions and make them visual
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title("Train")
plot_decision_boundary(model_1, X_train, y_train)
plt.subplot(1,2,1)
plt.title("Test")
plot_decision_boundary(model_1, X_test, y_test)
plt.show()

#^ Improving A Model (From a Model Perspective)
#? Add more layers - give the model more chances to learn more about patterns in the data
#? Add more hidden units - go from 5 hidden units to 10 hidden units
#? Fit for longer
#? Changing the activation functions
#? Change the learning rate
#? Change the loss function
