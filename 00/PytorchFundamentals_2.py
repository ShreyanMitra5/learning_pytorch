import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Tensor Manipulation

#Addition
tensor = torch.tensor([2, 3, 4])
tensor += 10
print(tensor)

#Multiplication
tensor *= 10
print(tensor)

#Pytorch Methods --> torch.mul(tensor, 10)  && torch.add(tensor, 5)

'''
Matrix Multiplication--2 rules:

1) Inner vals must be the same
   ex: (3, 2) @ (3, 2) --> Won't work
   ex: (3, 2) @ (2, 3) --> Will work
   ex: (2, 3) @ (2, 3) --> Will work

2) Resulting matrix has outer dims
   ex: (3, 2) @ (2, 3) --> (3, 3) Resulting dimensions of final matrix
'''

#Matrix mult:
newTorch = torch.matmul(tensor, tensor)
print(newTorch)
# "@" symbol can be used for matrix mult as well

#Transpose: Switch a dimensions of a tensor to avoid shape-based error in matrix mult
#tensor.T --> transposes a tensor

tensor_A = torch.tensor([[1, 2, 4], [3, 5, 6]])
tensor_B = torch.tensor([[5, 6, 8], [9, 10, 11]])
print("\ntensor_A: ", tensor_A, tensor_A.shape)
print("tensor_B transposed: ", tensor_B.T.shape)
print("Matrix Multiplication: ", torch.matmul(tensor_A, tensor_B.T), torch.matmul(tensor_A, tensor_B.T).shape)

#torch.mm --> short for matrix multiplication