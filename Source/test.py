import os
from skimage import io
import matplotlib.pyplot as plt
from skimage.filters.rank import median
import numpy as np

from skimage.filters import threshold_otsu


parent = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
img_path = os.path.join(parent, "High_Res_Input_Images", "DAPI_atubulin_pattern_all_1_001_R3D_D3D_PRJ_w435.tif")
image = io.imread(img_path)

fig, ax = plt.subplots(1, figsize = (10,10)) 
ax.imshow(image, cmap='gray', interpolation='nearest')
ax.set_aspect('equal')
plt.show()   




parent = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
img_path = os.path.join(parent, "High_Res_Input_Images", "DAPI_atubulin_pattern_all_1_001_R3D_D3D_PRJ_w435.tif")
image = io.imread(img_path)

std = np.std(image[:,:,2])
mean = np.mean(image[:,:,2])
thresh = threshold_otsu(image[:,:,2])

min_val = mean + std

image = np.clip(image[:,:,2], thresh - std, float('inf'))




fig, ax = plt.subplots(1, figsize = (10,10)) 
ax.imshow(image, cmap='gray', interpolation='nearest')
ax.set_aspect('equal')
plt.show()   