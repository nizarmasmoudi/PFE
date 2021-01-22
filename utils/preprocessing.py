import cv2
import pandas as pd
import os
import numpy as np

def split_image(img_path, overlap=0, show=False, save_output=None):
    img = cv2.imread(img_path)
    height, width = img.shape[:2]
    title = img_path.split('/')[-1].split('.')[0]
    pieces = [
        img[:height//2 + overlap, :width//2 + overlap, :], # Top left 
        img[height//2 - overlap:, :width//2 + overlap, :], # Bottom left
        img[:height//2 + overlap, width//2 - overlap:, :], # Top right
        img[height//2 - overlap:, width//2 - overlap:, :]  # Bottom right
    ]
    if show:
        window_titles = ['Top left piece', 'Bottom left piece', 'Top right piece', 'Bottom right piece']
        print('Original shape {}, split into 4 pieces of equal shapes {}'.format((height, width), (height//2 + overlap, width//2 + overlap)))
        for i, piece in enumerate(pieces):
            cv2.namedWindow(window_titles[i])
            cv2.moveWindow(window_titles[i], 40 if i<2 else 40+overlap+width//2, 30 if i%2==0 else 30+overlap+height//2)
            cv2.imshow(window_titles[i], piece)
            cv2.waitKey(0)
    if save_output:
        for i, piece in enumerate(pieces):
            cv2.imwrite(save_output + '/' + title + '_' + str(i+1) + '.jpg', piece)

def split_annotation(ann_path, overlap=0, save_output=None):
    img_path = ann_path.replace('annotations', 'images').replace('txt', 'jpg')
    height, width = cv2.imread(img_path).shape[:2]
    title = img_path.split('/')[-1].split('.')[0]
    annotations = pd.read_csv(ann_path, header=None)
    annotations = annotations[annotations[5].isin([1, 2])]
    
    bottom_right = []
    top_right = []
    bottom_left = []
    top_left = []
    for annotation in annotations.values:
        bbox_left, bbox_top, bbox_width, bbox_height, _, _, _, _ = annotation[:8]
        center = (bbox_left + bbox_width//2, bbox_top + bbox_height//2)
        if center[0] > (width//2 - overlap) and center[1] > (height//2 - overlap):
            bottom_right.append([max(bbox_left - (width//2 - overlap), 0), max(bbox_top - (height//2 - overlap), 0)] + list(annotation[2:]))
        if center[0] > (width//2 - overlap) and center[1] < (height//2 + overlap):
            top_right.append([max(bbox_left - (width//2 - overlap), 0)] + list(annotation[1:]))
        if center[0] < (width//2 + overlap) and center[1] > (height//2 - overlap):
            bottom_left.append([bbox_left, max(bbox_top - (height//2 - overlap), 0)] + list(annotation[2:]))
        if center[0] < (width//2 + overlap) and center[1] < (height//2 + overlap):
            top_left.append(list(annotation))
    if save_output:
        for i, split in enumerate([top_left, bottom_left, top_right, bottom_right]):
            pd.DataFrame(split).to_csv(save_output + '/' + title + '_{}.txt'.format(i+1), header=False, index=False)
            
def process_annotations(ann_path):
    output = []
    try:
        annotations = pd.read_csv(ann_path, header = None)
        annotations = annotations[annotations[5].isin([1, 2])].values
    except:
        return output
    img = cv2.imread(ann_path.replace('annotations', 'images').replace('.txt', '.jpg'))
    height_, width_ = img.shape[:2]
    for annotation in annotations:
        left, top, width, height, _, _, _, _ = annotation[:8]
        x_center = np.round((left + width/2) / width_, 5)
        y_center = np.round((top + height/2) / height_, 5)
        width = np.round(width/width_, 5)
        height = np.round(height/height_, 5)
        output.append(' '.join(['0', str(x_center), str(y_center), str(width), str(height)]))
    return output
        
def fill_ignored_regions(img_path, inplace=True, save_output=None):
    img = cv2.imread(img_path)
    try:
        annotations = pd.read_csv(img_path.replace('images', 'annotations').replace('.jpg', '.txt'), header = None).values
    except:
        annotations = []
    finally:
        if len(annotations) < 0:
            pass
        else:
            for left, top, width, height, _, obj, _, _ in annotations[:8]:
                if obj == 0:
                    cv2.rectangle(img, (left, top), (left+width, top+height), (230, 230, 230), -1) 
    if inplace:
        cv2.imwrite(img_path, img)
    elif save_output:
        cv2.imwrite(save_output + '/' + img_path.split('/')[-1], img)