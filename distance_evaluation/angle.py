import os
from utils import env
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

def vsplit(mat, n_splits):
    l = mat.shape[0]
    if l%n_splits > 0:
        arr = np.vsplit(mat[:-(l%n_splits), ...], n_splits)
        arr[-1] = np.append(arr[-1], mat[-(l%n_splits):, ...], axis=0)
        return arr
    else:
        return np.vsplit(mat, n_splits)

def heatmap(img_path, view=False, n_splits=3):
    ## Reading data
    annotation = img_path.replace('images', 'annotations').replace('.jpg', '.txt')
    img = plt.imread(img_path)
    df = pd.read_csv(annotation, header=None)
    df = df[df[5].isin([1, 2])]
    ## Calculating heatmap
    objects = {}
    for left, top, width, height, _, _, _, _ in df.values:
        objects[(left, top)] = width*height
    heatmap = np.zeros(img.shape[:2])
    for left, top in objects.keys():
        full_image = img.shape[0]*img.shape[1]
        heatmap[top:top+height, left:left+width] = objects[(left, top)]/full_image
        
    sub_heatmaps = vsplit(heatmap, n_splits)
    
    for sub_heatmap in sub_heatmaps:
        sub_heatmap[...] = np.max(sub_heatmap)
    
    heatmap = np.concatenate(sub_heatmaps, axis=0)
    heatmap = pd.DataFrame(heatmap).replace(to_replace=0, method='bfill').values
    ## Drawing
    if view:
        fig, (ax1, ax2) = plt.subplots(2, 1)
        ## Image
        ax1.imshow(img)
        ax1.set_title('Original image', fontsize='small')
        ## Heatmap
        m = ax2.imshow(heatmap, vmin=0, vmax=.05)
        cb = fig.colorbar(m, ticks=[0, .025, .05], ax=ax2)
        cb.outline.set_edgecolor('white')
        cb.ax.tick_params(size=0)
        cb.ax.set_yticklabels(['Small (inexistant)', 'Medium', 'Large'], fontsize='small') 
        for ax in [ax1, ax2]:
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.tick_params(bottom = False, top = False, left = False, right = False, 
                            labelbottom = False, labeltop = False, labelleft = False, labelright = False)
        ax2.set_title('Heatmap', fontsize='small')
        plt.show()
        
    return heatmap


images = ['0000022_00500_d_0000005', 
          '0000023_01233_d_0000011', 
          '0000086_01954_d_0000005', 
          '0000103_02964_d_0000030', 
          '0000253_00001_d_0000001']

for image in images:
    IMAGE = os.path.join(env.VALIDATION_PATH_, 'images', '{}.jpg'.format(image))
    heatmap_ = heatmap(img_path=IMAGE, view=True, n_splits=10)
    heatmap_[heatmap_ == 0] = np.mean(heatmap_[heatmap_ != 0])
    print(image, np.std(heatmap_)*1000)