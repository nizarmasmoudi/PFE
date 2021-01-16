import matplotlib.pyplot as plt
import pandas as pd
import os
from pprint import pprint

def split_image(image, overlap=0, plot=False, save_output=True):
    img = plt.imread(image)
    width, height = img.shape[:2]
    pieces = [
        img[:width//2 + overlap, :height//2 + overlap, :], # Top left 
        img[width//2 - overlap:, :height//2 + overlap, :], # Bottom left
        img[:width//2 + overlap, height//2 - overlap:, :], # Top right
        img[width//2 - overlap:, height//2 - overlap:, :]  # Bottom right
    ]
    
    if plot:
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3)
        ax1.imshow(pieces[0])
        ax1.set_title('Top left {}'.format(pieces[0].shape), size = 7, alpha = .8)
        ax7.imshow(pieces[1])
        ax7.set_title('Bottom left {}'.format(pieces[1].shape), size = 7, alpha = .8)
        ax3.imshow(pieces[2])
        ax3.set_title('Top right {}'.format(pieces[2].shape), size = 7, alpha = .8)
        ax9.imshow(pieces[3])
        ax9.set_title('Bottom right {}'.format(pieces[3].shape), size = 7, alpha = .8)
        ax5.imshow(img)
        ax5.set_title('Full image {}'.format(img.shape), size = 7, alpha = .8)
        for ax in [ax2, ax4, ax6, ax8]: ax.remove()
        for ax in [ax1, ax3, ax5, ax7, ax9]:
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.tick_params(which='both', 
                        bottom = False, top = False, left = False, right = False, 
                        labelbottom = False, labeltop = False, labelleft = False, labelright = False)
        fig.set_size_inches(18.5, 10.5)
        plt.show()
    if save_output:
        for i, piece in enumerate(pieces):
            plt.imsave('./draft/test_image_{}.jpg'.format(i), piece)
                
def split_annotation(annotation_file, overlap=0, save_output=True):
    height, width = plt.imread(annotation_file.replace('annotations', 'images').replace('txt', 'jpg')).shape[:2]
    annotations = pd.read_csv(annotation_file, header=None)
    annotations = annotations[annotations[5].isin([1, 2])]
    bottom_right = []
    top_right = []
    bottom_left = []
    top_left = []
    for annotation in annotations.values:
        bbox_left, bbox_top, bbox_width, bbox_height, _, _, _, _ = annotation
        center = (bbox_left + bbox_width//2, bbox_top + bbox_height//2)
        if center[0] > width//2 and center[1] > height//2:
            bottom_right.append([max(bbox_left - (width//2 - overlap), 0), max(bbox_top - (height//2 - overlap), 0)] + list(annotation[2:]))
        elif center[0] > width//2 and center[1] < height//2:
            top_right.append([max(bbox_left - (width//2 - overlap), 0)] + list(annotation[1:]))
        elif center[0] < width//2 and center[1] > height//2:
            bottom_left.append([bbox_left, max(bbox_top - (height//2 - overlap), 0)] + list(annotation[2:]))
        elif center[0] < width//2 and center[1] < height//2:
            top_left.append(list(annotation))
    if save_output:
        for i, split in enumerate([top_left, bottom_left, top_right, bottom_right]):
            pd.DataFrame(split).to_csv('./draft/test_image_{}.txt'.format(i), header=False, index=False)