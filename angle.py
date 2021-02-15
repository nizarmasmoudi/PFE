from dataset import Image
from utils.matrix import vsplit
from utils import env
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## * 90° angle : 0000333_01961_d_0000011
## * Low angle (garden) : 0000022_00500_d_0000005
## * Low angle (basketball) : 0000081_00000_d_0000001
## * Close capture : 0000103_02964_d_0000030

img_path = os.path.join(env.VALID_PATH, 'images', '0000022_00500_d_0000005.jpg')

def heatmap(img_path, view=False, return_std=True):
    image = Image(img_path)
    image.show(draw_bbox=True)
    
    arr = np.zeros((image.height, image.width))
    annotations = image.read_annotations()
    for left, top, width, height, _, _, _, _ in annotations.values[:, :8]:
        arr[top:top+height, left:left+width] = (width*height)*100/(image.height*image.width)
    splits = vsplit(arr, n_splits=7)
    for split in splits: split[:, :] = np.max(split)
    arr = np.concatenate(splits, axis=0)
    arr = pd.DataFrame(arr).replace(to_replace=0, method='ffill').values
    
    if view:
        plt.imshow(arr, cmap='Blues', vmin=0, vmax=.5)
        cbar = plt.colorbar(ticks=[0, .25, .5])
        cbar.outline.set_edgecolor('white')
        cbar.ax.tick_params(size=0)
        cbar.ax.set_yticklabels(['Small (inexistant)', 'Medium', 'Large'], fontsize='small') 
        plt.show()
        
    if return_std: return np.std(arr)


d = 1 # mètre
sigma = heatmap(img_path)
print((1 + sigma)*np.exp(d))
