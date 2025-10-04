import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Reshaping, stacking, squeezing, unsqueezing

x = torch.arange(1., 8.) #vals from 1, 8 (excluding), dim: 1

#Adds an extra dimension
x_reshaped = x.reshape(1, 7)
print(x_reshaped)

#torch.view() creates a new view of the og tensor, & any change applied to it changes og tensor

#stacking the same tensor on top of it several times
x_stacked = torch.stack([x, x, x], dim=0)
print(x_stacked)

#remove a single dimension of a tensor
print(f"Previous tensor: {x_reshaped}")
print(f"Previous shape: {x_reshaped.shape}")

# Remove extra dimension from x_reshaped
print(f"\nPrevious tensor: {x_reshaped}")
print(f"Previous shape: {x_reshaped.shape}")

x_squeezed = x_reshaped.squeeze()
print(f"\nNew tensor: {x_squeezed}")
print(f"New shape: {x_squeezed.shape}")

#x_reshaped.unsqueeze() will do the exact opposite

#We can rearrange the order of the axes thru .permute() method
x_rand = torch.rand(size=(224, 224, 3))
x_permuted = x_rand.permute(2, 0, 1) # shifts axis 0->1, 1->2, 2->0
print('\n', x_rand.shape)
print(x_permuted.shape) 


#Indexing
