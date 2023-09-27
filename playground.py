import torch
from model_classes.LinearRegressionModel import LinearRegressionModel
from utils.loadModel import loadModel
from utils.loadModelFromStateDict import loadModelFromStateDict

model = loadModel('04_pytorch_workflow_model.pt', LinearRegressionModel)
model2 = loadModelFromStateDict('04_pytorch_workflow_state_dict.pt', LinearRegressionModel)
print(model.state_dict())
print(model2.state_dict())