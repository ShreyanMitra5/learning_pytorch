import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#print(torch.__version__)

#tensors are the building components in torch--numeric data representation

#scalar object initalized
scalar = torch.tensor(7)
print(scalar)

#dimension of object
dim = scalar.ndim
print("dimension: ", dim)

#retrieve delcared that is set to tensor
print("value of item: ", scalar.item())