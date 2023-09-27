import sys
sys.path.insert(0, "/home/jc/Desktop/Projects/Python/FCC-DeepLearning")
# ^ Classification Exercise
# ? 1 Make a binary classification dataset with Scikit-Learn's make_moons() function
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from utils.printTensor import printTensor
from matplotlib import pyplot as plt
import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS = 500
N_SAMPLES = (500, 500)
NOISE = 0.03
RANDOM_SEED = 85
MOON_IN_FEATURES = 2
MOON_OUT_FEATURES = 1
MOON_HIDDEN_LAYERS = 32
LEARNING_RATE = 0.1

X, y = make_moons(
    n_samples=N_SAMPLES, shuffle=True, noise=NOISE, random_state=RANDOM_SEED
)

# plt.style.use("dark_background")
# plt.title("Moons")
# plt.scatter(x=X[:, 0], y=X[:, 1], c=y, cmap=plt.cm.RdYlBu)
# plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# ? 2. Build a model by subclassing nn.Module that incorporates non-linear activation functions and is capable of fitting the data you created in 1
class MoonModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features=MOON_IN_FEATURES, out_features=MOON_HIDDEN_LAYERS),
            nn.Linear(in_features=MOON_HIDDEN_LAYERS, out_features=MOON_HIDDEN_LAYERS),
            nn.ReLU(),
            nn.Linear(in_features=MOON_HIDDEN_LAYERS, out_features=MOON_HIDDEN_LAYERS),
            nn.Linear(in_features=MOON_HIDDEN_LAYERS, out_features=MOON_OUT_FEATURES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


moonModel = MoonModel()
moonModel.to(device)

X_train = torch.from_numpy(X_train).to(device).type(torch.float32)
X_test = torch.from_numpy(X_test).to(device).type(torch.float32)
y_train = torch.from_numpy(y_train).to(device).type(torch.float32)
y_test = torch.from_numpy(y_test).to(device).type(torch.float32)

printTensor(X_train, "X_train")

# ? 3. Setup a binary classification compatible loss fucntion and optimizer to use when training the model

loss_function = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params=moonModel.parameters(), lr=LEARNING_RATE)

# ? 4. Create a training and testing loop to fit the model you created in 2 to the data you created in 1
# ? 4.1 : Create an accuracy function using torchMetrics
# ? 4.2 Train the model for long enough for it to reach over 96% accuracy
# ? 4.3 Output progress every 10 epochs
from torchmetrics.classification import BinaryAccuracy

binary_acc_fn = BinaryAccuracy().to(device)

for epoch in range(EPOCHS):
    moonModel.train()
    y_pred_logits = moonModel(X_train).squeeze()
    
    y_pred = torch.round(torch.sigmoid(y_pred_logits))
    loss = loss_function(y_pred_logits, y_train)
    acc = binary_acc_fn(y_pred, y_train) * 100
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    moonModel.eval()
    with torch.inference_mode():
        test_logits = moonModel(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))
        test_loss = loss_function(test_logits, y_test)
        test_acc = binary_acc_fn(test_pred, y_test) * 100
        if epoch % 10 == 0:
            print(
                f"Epoch: {epoch} | Loss: {loss:.5f} | Acc: {acc:.2f}% | Test_Loss: {test_loss:.5f} | Test_Acc: {test_acc:.2f}"
            )

#? 5. Make Predictions with your trained model and plot them using the plot_descision_boundary() function

from utils.helperFunctions import plot_decision_boundary
plt.style.use("dark_background")
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title("Train")
plot_decision_boundary(moonModel, X_train, y_train)
plt.subplot(1,2,1)
plt.title("Test")
plot_decision_boundary(moonModel, X_test, y_test)
plt.show()