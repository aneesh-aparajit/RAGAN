import torch
import torch.nn as nn


'''
y     - input age
y'    - output age

x     - input image  
x'    - age transformed image -> G(x, y')
x_rec - G(x, y)
r_cyc - G(x', y)
'''