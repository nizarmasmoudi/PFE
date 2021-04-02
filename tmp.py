import os
import pandas as pd
import cv2 as cv

img_paths = [os.path.join('trash', 'images', img_path) for img_path in os.listdir(os.path.join('trash', 'images'))]

for img_path in img_paths:
    mat = cv.imread(img_path)
    parsed = pd.read_csv(img_path.replace('images', 'annotations').replace('.jpg', '.txt'), header=None, names=['object_class', 'x_center', 'y_center', 'width', 'height'], sep=' ')
    for obj, x_center, y_center, width, height in parsed.values:
        height = int(height*mat.shape[0])
        width = int(width*mat.shape[1])
        top = int(y_center*mat.shape[0] - height/2)
        left = int(x_center*mat.shape[1] - width/2)
        if obj == 0:
            cv.rectangle(mat, (left, top), (left+width, top+height), (0, 255, 0), 1)
        elif obj == 1:
            cv.rectangle(mat, (left, top), (left+width, top+height), (0, 0, 255), 1)
    cv.imshow('', mat)
    cv.waitKey(0)
    cv.destroyAllWindows()