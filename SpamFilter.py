import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load data and visualize it
mat_path = "./spamdata.mat"
data = sio.loadmat(mat_path)

# import variables
X = data.get('X')
y = data.get('y')
X2 = data.get('X2')
y2 = data.get('y2')