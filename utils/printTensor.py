def printTensor(tensor, tensorName: str):
  print(tensorName + ":\n", 
        tensor, 
        "\n", 
        "Dimensions: ", 
        tensor.ndim, 
        "\n", 
        "Size: ", 
        tensor.size(), 
        "\n", 
        "Datatype: ", 
        tensor.dtype, 
        "\n", 
        "Device: ", 
        tensor.device, 
        "\n\n")