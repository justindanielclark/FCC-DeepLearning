## 00. PyTorch Fundamentals
# Resource notebook: https://www.learnpytorch.io/00_pytorch_fundamentals/

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print(torch.__version__, "\n\n")

def printTensor(tensor, tensorName: str):
  print(tensorName + ":\n", tensor, "\n", "Dimensions: ", tensor.ndim, "\n", "Size: ", tensor.size(), "\n", "Datatype: ", tensor.dtype, "\n\n")
def printHeader(text: str):
  print("\n\n---" + text + "---\n")

#^ Introduction to Tensors
## The Main Building Block of Machine Learning
## A way to represent multidimensional data

##! Creating Tensors
printHeader("CREATING TENSORS")
# scalar
scalar = torch.tensor(7)
printTensor(scalar, "scalar")

# vector
vector = torch.tensor([7,7])
printTensor(vector, "vector")

# MATRIX
matrix = torch.tensor([[1,1,1],[2,2,2]])
printTensor(matrix, "matrix")

# TENSOR
TENSOR = torch.tensor([
  [
    [1,2,3], [4,5,6], [7,8,9]
  ], [
    [1,2,3], [4,5,6], [7,8,9]
  ], [
    [1,2,3], [4,5,6], [7,8,9]
  ]
])
printTensor(TENSOR, "TENSOR")
print(TENSOR[0], "\n\n")

##! Random Tensors
printHeader("RANDOM TENSORS")
# Random tensors are important because the way many neural networks learn is that they start with tensors
# full of random numbers, and then adjsut those random numbers to better represent the data.

# Start with random data -> look at data -> update random numbers -> look at data -> update random numbers

random_tensor = torch.rand(2,3,2,3)
printTensor(random_tensor, "random_tensor")

# Create a random tensor with similar shape to an image tensor
random_image_size_tensor = torch.rand(size=(224,224,3)) #height, width, color channels
random_image_size_tensor = torch.rand(size=(3,224,224)) #color channels, height, width

##! Zeros and Ones
printHeader("ZEROS AND ONES TENSORS")
zero_tensor = torch.zeros(size=(2,2))
printTensor(zero_tensor, "zero_tensor")

ones_tensor = torch.ones(size=(1,3,9))
printTensor(ones_tensor, "ones_tensor")

##! Creating A Range of Tensors
printHeader("A RANGE TENSORS")
print(torch.arange(0, 10)) ## Non Inclusive of End, Default Step 1
print(torch.arange(0,10.1))
print(torch.arange(start=0, end=10, step=1, dtype=torch.float16))
print(torch.arange(start=0, end=10, step=2))
print("\n\n")

##! Making A Tensor Like Another Tensor
printHeader("A LIKE TENSORS")
like_tensor = torch.tensor([1,2,3,4])
printTensor(like_tensor, "like_tensor")
zeroes_like = torch.zeros_like(like_tensor) ## Same shape as above, but all 0s
printTensor(zeroes_like, "zeroes_like")
ones_like = torch.ones_like(like_tensor) ## Same shape as above, but all 1s
printTensor(ones_like, "ones_like")

#^ DataTypes
printHeader("DATATYPES")
# One of the big sources of errors when using PyTorch/Deep Learning
standard_tensor = torch.tensor([1,2],
                               dtype=None,  ## Default: "torch.float32", What precision is it stored?
                               device=None, ## Default: "cpu", What device your tensor is on?
                               requires_grad=False ## Default: false, Should we track it over time?
)
printTensor(standard_tensor, "standard_tensor")

float_32_tensor = torch.tensor([1,2], dtype=torch.float32)
printTensor(float_32_tensor, "float_32_tensor")

float_16_tensor_conversion = float_32_tensor.type(torch.float16)
printTensor(float_16_tensor_conversion, "float_16_tensor_conversion")


#? Three Major Error Sources In PyTorch/Deep Learning
  #? 1.) Wrong Datatype
  #? 2.) Wrong Shape
  #? 3.) Wrong Device


#^ Manipulating Tensors / Tensor Operations
printHeader("MANIPULATING TENSORS")
#? Tensor Operations Include:
  #? Addition
  #? Subtraction
  #? Multiplication (Element-Wise)
  #? Division
  #? Matrix Multiplication

operation_tensor = torch.tensor([1,2,3])
printTensor(operation_tensor, "Tensor")

#! Adding Tensors
printHeader("ADDING TENSORS")
printTensor(operation_tensor + 10, "Tensor + 10")
printTensor(torch.add(operation_tensor, 10), "torch.add(tensor, 10)")
printTensor(torch.add(operation_tensor, operation_tensor), "torch.add(tensor, tensor)")
#! Subtracting Tensors
printHeader("SUBTRACTING TENSORS")
printTensor(operation_tensor - 10, "Tensor - 10")
printTensor(torch.sub(operation_tensor, 10), "torch.sub(tensor, 10)")
printTensor(torch.subtract(operation_tensor, operation_tensor), "torch.sub(tensor, tensor)")
#! Multiplying Tensors
printHeader("MULTIPLYING TENSORS")
printTensor(operation_tensor * 10, "Tensor * 10")
printTensor(torch.mul(operation_tensor, 10), "torch.mul(tensor, 10)")
printTensor(torch.mul(operation_tensor, operation_tensor), "torch.mul(tensor, tensor)")
#! Dividing Tensors
printHeader("DIVIDING TENSORS")
printTensor(operation_tensor / 10, "Tensor / 10")
printTensor(torch.div(operation_tensor, 10), "torch.divide(tensor, 10)")
printTensor(torch.div(operation_tensor, operation_tensor), "torch.div(tensor, tensor)")
#! Multiplying Matrixes
printHeader("MATMUL TENSORS")
#? https://www.mathsisfun.com/algebra/matrix-multiplying.html
#? Also known as the Dot Product
#? In General: (m x n) * (n x p) -> (m x p)
#? The Y of M1 must be equal in size to the X of M2
#? The Commutative Law of Multiplcation does not transfer to matrixes: a x b !== b x a

printTensor(operation_tensor * operation_tensor, "Tensor * Tensor")

printTensor(torch.matmul(operation_tensor, operation_tensor), "torch.matmul(Tensor, Tensor)")

#? In General: torch.operation is signficantly faster than the operation equivalent

#? Two main rules that performing matrix multiplcation needs to satisfy
  #? 1. The 'inner' dimensions of the tensors must match 
    #? (3,2) @ (3,2) wont work
    #? (3,2) @ (2,3) will work

#^ Fixing a Shape Issue
printHeader("ADDRESSING SHAPE")
#? To Fix a tensor shape issue, we can manipulate the shape of one of our tensors using a transpose

tensor_A = torch.tensor([[1,2],[3,4],[5,6]])
tensor_B = torch.tensor([[-1,-2],[-3,-4],[-5,-6]])

try:
  torch.matmul(tensor_A, tensor_B)
except:
  print("Issue due to tensor shape!, cannot multiply shape 3x2 by 3x2")

transposed_tensor_B = tensor_B.T
#? tensor.T flips a matrix dimensions, tensor_B becomes
#? [[-1, -3, -5], [-2, -4, -6]]

printTensor(transposed_tensor_B, "transposed_tensor_B")

#^ Tensor Basic Math Functions
printHeader("BASIC MATH FUNCTIONS")
#? Largest Individual Scalar In A Tensor
print(torch.max(tensor_A))
print(tensor_A.max())
#? Largest Tensor
print(torch.max(tensor_A, tensor_B))
#? Sum
print(torch.sum(tensor_A))
#? What Index Is The Largest Value At
print(torch.argmax(tensor_A))

#^ Reshaping, stacking, squeezing, and unsqueezing tensors
printHeader("RESHAPING, STACKING, SQUEEZING, AND UNSQUEEZING")
#? Reshaping - reshapes an input tensor to a defined shape
#? View - returns a view of an input tensor of a certain shape, but keeps the same memory as the original tensor
#? Stacking - combine multiple tensors on top of each other (vstack) or side by side (hstack)
#? Squeeze - removes all 1 dimensions from a tensor
#? Unsqueeze - adds a 1 dimension to target tensor
#? Permute - Return a view of the input with dimensions permuted (swapped) in a certain way

printHeader("RESHAPING")
printTensor(tensor_A, "tensor_A")
printTensor(tensor_A.reshape(2,3), "tensor_A.reshape(2,3)") 
printTensor(tensor_A.reshape(1,6), "tensor_A.reshape(1,6)") 
printTensor(tensor_A.reshape(6,1), "tensor_A.reshape(6,1)") 
try:
  tensor_A.reshape(3,5)
except:
  print("Cannot change shape to a different size than original")

printHeader("VIEW")
printTensor(tensor_A.view(size=[1,6]), "tensor_A.view(size=[1,6])")
printTensor(tensor_A.view(size=[6]), "tensor_A.view(size=[6])")

printHeader("STACKING")
printTensor(torch.stack([tensor_A, tensor_B, tensor_A, tensor_B]), "torch.stack([tensor_A, tensor_B, tensor_A, tensor_B])")
printTensor(torch.stack([tensor_A, tensor_B], dim=1), "torch.stack([tensor_A, tensor_B], dim=1)")
printTensor(torch.stack([tensor_A, tensor_B], dim=2), "torch.stack([tensor_A, tensor_B], dim=2)")
printTensor(torch.stack([tensor_A, tensor_B], dim=-1), "torch.stack([tensor_A, tensor_B], dim=-1)")
printTensor(torch.stack([tensor_A, tensor_B], dim=-2), "torch.stack([tensor_A, tensor_B], dim=-2)")
printTensor(torch.stack([tensor_A, tensor_B], dim=-3), "torch.stack([tensor_A, tensor_B], dim=-3)")

printHeader("SQUEEZING")
x = torch.rand(2,1,2)
printTensor(x, "x = torch.rand(2,1,2)")
y = torch.squeeze(x)
printTensor(y, "y = torch.squeeze(x)")
y = torch.squeeze(x, 1)
printTensor(y, "y = torch.squeeze(x, 1)")

printHeader("UNSQUEEZING")
x = torch.rand(2,2,2)
printTensor(x, "x = torch.rand(2,2,2)")
y = torch.unsqueeze(x, 0)
printTensor(y, "y = torch.unsqueeze(x, 0)")
y = torch.unsqueeze(x, -1)
printTensor(y, "y = torch.unsqueeze(x, -1)")

printHeader("PERMUTE")
imageData = torch.rand(224,224,3) #? 224 x 224 image with 3 color channels
imageDataPermute = imageData.permute(2,0,1) #? 3 color Channels, then the 224 x 224 image

print("imageData.shape", imageData.shape)
print("imageDataPermute.shape", imageDataPermute.shape)

#^ Indexing
printHeader("INDEXING")
x = torch.arange(1, 10).reshape(1,3,3)
printTensor(x, "x = torch.arange(1, 10).reshape(1,3,3)")

printTensor(x[0], "x[0]")
printTensor(x[0][0], "x[0][0]")
printTensor(x[0][0][2], "x[0][0][2]")

try:
  printTensor(x[0][0][3], "x[0][0][3]")
except:
  print("Out of Bounds!\n")

try:
  printTensor(x[1][0][0], "x[1][0][0]")
except:
  print("Out of Bounds!\n")

printTensor(x[0 , 2, 2], "x[0,2,2]")

#? Get all values of the 0th and 1st dimensions, but only index 1 of the 2nd dimension
printTensor(x[:,:,1], "x[:,:,1]")

#? Get all values of the 0th dimension, but only the 1 index value of the 1st and 2nd dimension
printTensor(x[:,1,1], "x[:,1,1]")

#? Get index 0 of 0th and 1st dimension, and all values of 2nd dimension
printTensor(x[0,0,:], "x[0,0,:]")

#? Get 9
printTensor(x[:,2,2], "Get 9: x[:,2,2]")

#? Get 3, 6, 9 |OR| get all of the elements of the 2nd dimension
printTensor(x[:,:,2], "Get 3, 6, 9 |OR| get all of the elements of the 2nd dimension, x[:,:,2]")

#^ PyTorch Tensors and NumPy
#? NumPy is a popular scientific Python numerical computing library
#? PyTorch has functionality to interact with it
#? torch.from_numpy(ndarray)

np_array = np.arange(1.0, 8.0) # [1. 2. 3. 4. 5. 6. 7.], dtype=float64
tensor = torch.from_numpy(np_array) # [1. 2. 3. 4. 5. 6. 7.] dtype=float64
tensor_w_modified_type = torch.from_numpy(np_array).type(torch.float32) # [1. 2. 3. 4. 5. 6. 7.] dtype=float32

tensor = torch.ones(7) # [1,1,1,1,1,1,1], dtype=float32
numpy_tensor = tensor.numpy() #[1,1,1,1,1,1,1], dtype=float32

#? Default datatype of numpy is float64, default datatype of pytorch is float32.