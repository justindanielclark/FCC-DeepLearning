
#^ PyTorch Workflow

#^ https://www.learnpytorch.io/01_pytorch_workflow/

#^ Overview
#? 1. Get Data Ready (turn it into tensors)
#? 2. Build or Pick A Pretrained Model (to suit problem)
  #? 2.1 Pick a loss function and optimizer
  #? 2.2 Build a training loop
#? 3. Fit the model to the data and make a prediction
#? 4. Evaluate the model
#? 5. Improve through Experimentation
#? 6. Export Model

import torch
from torch import nn ##contains all of PyTorch's building blocks for NeuralNetworks
#? torch.nn = https://pytorch.org/docs/stable/nn.html
from utils.printTensor import printTensor as pt
from utils.plotPredictions import plot_predictions as pp

#^ Data (Preparing and Loading)
#? Data can be almost anything in machine learning
    #? Convert Data into numerical representation
#? Lets create some known data using the linear regression formula (y = a + bx)

#? We will use the linear regression formula to make a straight line with known parameters
start = 0
end = 1
step = .02
#? THE INPUT VALUES
X = torch.arange(start, end, step).unsqueeze(dim=1) 
#? ITS LABELING, AS A DEFINED KNOWN RELATIONSHIP (TO US).
#? Normally, we don't know what the relationship is, we just know that x is mapped to some value y. In this case, we explicitly know how to go from x -> y
weight = .7           #? b in our linear regression
bias = .3             #? a in our linear regression
y = weight * X + bias #? y, our output

pt(X[:10], "X[:10]")
pt(y[:10], "y[:10]")

#^ Splitting data into training and test sets
#! Arguably the most important concept in handling data in Machine Learning
#? We need 3 sets of data:
  #? Training Set
    #? The model learns from this data (like the course materials you study during the semester)
    #? Amount of total data: (~60-80)%
    #? How Often is it used? Always
  #? Validation Set
    #? The model gets tuned on this data (like a practice exam)
    #? Amount of total data: (~10-20)%
    #? How Often is it used? Often
  #? Testing Set
    #? The Model gets evaluated on this data to test what it has learned (like a final exam)
    #? Amount of total data: (~10-20)%
    #? How Often is it used? Always

#? Creating a Training/Testing Split
train_split = int(.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

#^ Visualizing Data
#! See utils plot_predictions, turn below comment off if you want to see the plot
#TODO pp(X_train, y_train, X_test, y_test, None)

#^ Build Model
#? Create a linear regression model class
class LinearRegressionModel(nn.Module): #! Nearly everything in PyTorch inherits from nn.Module
  def __init__(self):
    super().__init__()
    #! Declare All Model Parameters
    self.weights = nn.Parameter(torch.randn(1,                    # Start with a random weight, try to adjust it to ideal weight
                                            requires_grad=True,   # <- Can this be updated via gradient descent (default is true)
                                            dtype=torch.float))   # Our datatype
    self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
  #! Forward Method to define the computation in the model, must be overwritten
  def forward(self, x: torch.Tensor) -> torch.Tensor: # x is our input data
    return self.weights * x + self.bias # Our linear regression

#? What our model does:
  #? Start with random values (weight and bias)
  #? Look at training data and adjust the random values to better represent (or get closer to) the ideal values (the weight and bias values we used to create the data)

#? How does it do so? Two Main Algorithms (PyTorch does this for us)
  #? Gradient Descent https://youtu.be/IHZwWFHWa-w
  #? Backpropagation https://youtu.be/llg3gGewQ5U

#^ PyTorch Model Building Essentials
#? torch.nn - contains all the building blocks for computations graphs (a neural network can be considered a computational graph)
#? torch.nn.Parameter - Stores tensors that can be used with nn.Module. if requires_grad=True gradients (used for updating model parameters via gradient descent) are calculated automatically. This is often referred to as autograd
#? torch.nn.Module - The base class for all neural network modules, all the building blocks for neural networks are subclasses. If you are building a nn in PyTorch, your modles should be a subclass of nn.Module. Requires a forward() method to be implemented
#? torch.optim - Contains various optimization algos (these tell the model parameters stored in nn.Parameter how to best change to improve gradient descent and in turn reduce loss)
#? def forward() - All nn.Module classes require a forward() method. this defines the computation that will take place ont eh data passed to the particular nn.Module

#^ Checking the Contents of our PyTorch Model
#? Now That We've Created A Model, Let's See Whats Inside...
#? model.parameters() will show a list of existing parameters...

# RANDOM_SEED = 42
# torch.manual_seed(RANDOM_SEED)

model_0 = LinearRegressionModel()
# print(list(model_0.parameters()))
print(model_0.state_dict())


#^ Make Predictions Using `torch.inference_mode()`
#? To Check our model's predictive power, lets see how well it predicts `y_test` based on `X_test`
#? When we pass data through our model, it will run it through the forward() method

# y_pred = model_0(X_test) # Stores Gradient Information (additional data)

# with torch.inference_mode(): # Does Not Store Gradient Information, faster on one offs
#   y_pred = model_0(X_test)

with torch.no_grad(): # Also Does not Store Gradient Information
  y_pred = model_0(X_test)

#? The whole idea of training is for some model to move from some *unknown* parameters to a *ideal* parameters
#! The way to move from poor->ideal parameters is to use loss functions
#! Loss Functions may also be called "cost functions" | "criterion" in different areas

#^ Loss Functions and Optimizers
#? Loss Functions: A function to measure the difference between your model's predictions and its ideal outputs. It gives a number representation of how wrong your model's predictions are
#? Optimizer: Take into account the loss of a model and adjusts the model's parameters (eg weights & bias) to improve the loss function
#? For PyTorch, to implement the above, we will need to implement a:
  #? A Training Loop
  #? A Testing Loop

#? Setup Loss Function
loss_fn = nn.L1Loss() #MAE - Mean Absolute Error / difference between y_test and y_pred

#? Setup Optimizer
optimizer = torch.optim.SGD(
  params=model_0.parameters(), # Our Models Parameters
  lr=0.01 # Learning Rate: Possibly the most important hyperparameter you can set, default is 0.0001, the smaller the lr, the smaller the changes in the parameter to attempt to diminish loss
)

#? Q: Which loss function and optimizer should I use?
#? A: This is problem specific. With experience, you'll gain an idea of what works and what doesn't with your particular problem set.
  #? For regression based problems, a loss function like MAE and an optimizer like SGD suffice
  #? For classification problems, BCELoss() tend to perform significantly better

#^ Setup A Training Loop
#? We Need A Couple things
  #? 0. Loop through the data and do the following...
  #? 1. forward pass( this involves data moving through our model's forward function, also called forward propagation) to make predictions on data
  #? 2. Calculate the loss (compare foward pass predictions to ground truth labels)
  #? 3. Optimizer zero grad
  #? 4. Loss backwards - move backwards through the network to calculate the gradients of eachof the parameters of our model with respect to the loss
  #? 5. Optimizer Step - use the optimizer to adjust our model's parameters to try and improve the loss

#? An Epoch is one loop through the data...
  #? Epoch is a hyperparameter because we have set it ourself

pp(X_train, y_train, X_test, y_test, y_pred)

epochs = 1000
print(list(model_0.parameters()))
#^ Training...
#? 0. Loop through the data
for epoch in range(epochs):
  model_0.train() # train mode requires all models that require gradients to require gradients
  #? 1. Forward Pass
  y_pred = model_0(X_train)
  #? 2. Calculate Loss - loss(input, target)
  loss = loss_fn(y_pred, y_train)
  #? 3. Optimizer Zero Grad
  optimizer.zero_grad()
  #? 4. Perform Backpropagation on the loss with respect to the parameters of the model
  loss.backward()
  #? 5. Step the optimizer (perform gradient descent)
  optimizer.step() # my default, the optimizer changes will acculmulate through the loop so we have to zero them above in step 3

  #^Test Time...
  model_0.eval() # Turns off settings not needed for testing...
  with torch.inference_mode(): # Turn on inference mode context manager (removes even more things not needed for inference)
    #? 1. Do the forward pass
    test_pred = model_0(X_test)
    #? 2. Calculate the Loss
    test_loss = loss_fn(test_pred, y_test)
  #^ How We Doing??
  print(f"Epoch: {epoch} | Train loss: {loss:.4f} | Test loss: {test_loss:.4f}")

print(list(model_0.parameters()))
pp(X_train, y_train, X_test, y_test, test_pred)

#^ Training Loop 2 With A Finer Tuned Learning Rate...

optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.001 )

for epoch in range(epochs):
  model_0.train()
  y_pred = model_0(X_train)
  loss = loss_fn(y_pred, y_train)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  model_0.eval()
  with torch.inference_mode():
    test_pred = model_0(X_test)
    test_loss = loss_fn(test_pred, y_test)
  print(f"Epoch: {epoch} | Train loss: {loss:.4f} | Test loss: {test_loss:.4f}")

print(list(model_0.parameters()))
pp(X_train, y_train, X_test, y_test, test_pred)


#^ Training Loop 3 With An Even Finer Tuned Learning Rate...
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.0001 )

for epoch in range(epochs):
  model_0.train()
  y_pred = model_0(X_train)
  loss = loss_fn(y_pred, y_train)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  model_0.eval()
  with torch.inference_mode():
    test_pred = model_0(X_test)
    test_loss = loss_fn(test_pred, y_test)
  print(f"Epoch: {epoch} | Train loss: {loss:.4f} | Test loss: {test_loss:.4f}")

print(list(model_0.parameters()))
pp(X_train, y_train, X_test, y_test, test_pred)


#^ Training Loop 4 With An Even Finer Tuned Learning Rate...
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.000001 )

for epoch in range(epochs):
  model_0.train()
  y_pred = model_0(X_train)
  loss = loss_fn(y_pred, y_train)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  model_0.eval()
  with torch.inference_mode():
    test_pred = model_0(X_test)
    test_loss = loss_fn(test_pred, y_test)
  print(f"Epoch: {epoch} | Train loss: {loss:.4f} | Test loss: {test_loss:.4f}")

print(list(model_0.parameters()))
pp(X_train, y_train, X_test, y_test, test_pred)
