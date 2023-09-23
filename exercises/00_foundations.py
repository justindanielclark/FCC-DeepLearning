
import torch
import sys
sys.path.insert(0, "/home/jc/Desktop/Projects/Python/FCC-DeepLearning")
from utils.printTensor import printTensor as pt

#^ EXERCISES: https://www.learnpytorch.io/00_pytorch_fundamentals/#exercises

#? Create a random tensor with shape (7,7)
tensorA = torch.rand([7,7])
pt(tensorA, "torch.rand([7,7])")

#? Perform a matrix multiplication on the tensor from 2 with another random tensor with shape (1, 7) (hint: you may have to transpose the second tensor).
tensorB = torch.rand([1,7])
result = torch.matmul(tensorA, tensorB.T)
pt(result, "torch.matmul(tensorA, (torch.rand([1,7])).T)")

#? Set the random seed to 0 and do exercises 2 & 3 over again.
RANDOM_SEED = 0
torch.manual_seed(RANDOM_SEED)
tensorA = torch.rand([7,7])
torch.manual_seed(RANDOM_SEED)
tensorB = torch.rand([1,7])
result = torch.matmul(tensorA, tensorB.T)
pt(result, "torch.matmul(tensorA, (torch.rand([1,7])).T)")

#? Speaking of random seeds, we saw how to set it with torch.manual_seed() but is there a GPU equivalent? 
#? (hint: you'll need to look into the documentation for torch.cuda for this one). If there is, set the GPU random seed to 1234.

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.manual_seed(RANDOM_SEED)
tensorA = torch.rand([7,7], device=device)
pt(tensorA, "tensorA on GPU, manual seed 1 on cuda")

torch.cuda.manual_seed(RANDOM_SEED)
tensorA = torch.rand([7,7], device=device)
pt(tensorA, "tensorA on GPU, manual seed 2 on cuda")

#? Create two random tensors of shape (2, 3) and send them both to the GPU 
#? (you'll need access to a GPU for this). Set torch.manual_seed(1234) when creating the tensors (this doesn't have to be the GPU random seed).
RANDOM_SEED = 1234
torch.manual_seed(RANDOM_SEED)
tensorA = torch.rand([2,3], device="cuda")
pt(tensorA, "tensorA on GPU, manual seed w/o manual seed declaration on cuda")
torch.manual_seed(RANDOM_SEED)
tensorB = torch.rand([2,3], device="cuda")
pt(tensorB, "tensorB on GPU, manual seed w/o manual seed declaration on cuda")


#? Perform a matrix multiplication on the tensors you created in 6 (again, you may have to adjust the shapes of one of the tensors).
result = torch.matmul(tensorA, tensorB.T)
pt(result, "torch.matmul(tensorA, tensorB.T)")

#? Find the maximum and minimum values of the output of 7.
#? Find the maximum and minimum index values of the output of 7.
pt(result.max(), "result.max()")
pt(result.min(), "result.min()")
print(result.argmax())
print(result.argmin())