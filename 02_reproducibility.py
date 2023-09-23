
import torch
import numpy as np
from utils.printTensor import printTensor as pt
from utils.printHeader import printHeader as ph

#! https://pytorch.org/docs/stable/notes/randomness.html
#! https://en.wikipedia.org/wiki/Random_seed

#^ Reproducibility (trying to take the random out of random)
random_tensor_A = torch.rand([3,3], dtype=torch.float16)
ph("A Random Tensor")
pt(random_tensor_A, "random_tensor_A = torch.rand([3,3], dtype=torch.float16)")

#? Setting a random seed
RANDOM_SEED = 42 
torch.manual_seed(RANDOM_SEED)

ph("Random Tensors After Setting Seed = 42")
random_tensor_B = torch.rand([3,3], dtype=torch.float16)
pt(random_tensor_B, "random_tensor_B = torch.rand([3,3], dtype=torch.float16)")
random_tensor_C = torch.rand([3,3], dtype=torch.float16)
pt(random_tensor_C, "random_tensor_C = torch.rand([3,3], dtype=torch.float16)")

ph("Random Tensors After Setting Seed = 42 Before Each .rand call")
torch.manual_seed(RANDOM_SEED)
random_tensor_D = torch.rand([3,3], dtype=torch.float16)
pt(random_tensor_D, "random_tensor_D = torch.rand([3,3], dtype=torch.float16)")
torch.manual_seed(RANDOM_SEED)
random_tensor_E = torch.rand([3,3], dtype=torch.float16)
pt(random_tensor_E, "random_tensor_E = torch.rand([3,3], dtype=torch.float16)")

