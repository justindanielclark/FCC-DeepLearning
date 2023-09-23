
import torch
from utils.printTensor import printTensor as pt
from utils.printHeader import printHeader as ph

#! GPUs = faster computation on numbers, thanks to CUDA + NVIDIA hardward + PyTorch working behind the scenes
#? Getting a GPU
  #? 1. Easiest - Use Google Colab for a free GPU (options to upgrade)
  #? 2. Use your own GPU
  #? 3. Use cloud computing - GCP, AWS, Azure

#! Checking if GPU/CUDA is available:
print(torch.cuda.is_available())
# In Terminal: 
## nvcc --version
## nvidia-smi

#! Setting Up Device Agnostic Code (Simplistic)
device = "cuda" if torch.cuda.is_available() else "cpu"
tensor_on_default_device = torch.rand([2,2], device=device, dtype=torch.float16)
pt(tensor_on_default_device, "tensor_on_default_device")

tensor_on_cpu = tensor_on_default_device.to("cpu")
pt(tensor_on_cpu, "tensor_on_cpu")
pt(tensor_on_default_device, "tensor_on_defualt_device")

tensor_on_gpu = tensor_on_cpu.to("cuda")
pt(tensor_on_gpu, "tensor_on_gpu")

#! Not Allowed To Take GPU Tensors and put them into NumPy, must convert first
try:
  tensor_on_gpu.numpy()
except:
  print('Not Allowed\n')

pt(tensor_on_gpu.cpu(), "tensor_on_gpu.cpu()")

cpu_from_gpu = tensor_on_gpu.cpu()
pt(cpu_from_gpu, "cpu_from_gpu")
print("cpu_from_gpu.numpy()", cpu_from_gpu.numpy(), "\n")
cpu_from_gpu_64 = cpu_from_gpu.type(torch.float64)

pt(cpu_from_gpu, "cpu_from_gpu")

pt(cpu_from_gpu_64, "cpu_from_gpu_64")

print("cpu_from_gpu_64.numpy()", cpu_from_gpu_64.numpy(), "\n")