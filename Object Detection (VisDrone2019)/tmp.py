from dataset.processing import Image, read_image
import dataset.navigation as nav
import pandas as pd
import cv2 as cv
import os

# img_path = os.path.join(nav.VALID_PATH, 'images', '0000291_03601_d_0000886.jpg')
img_path = '10.jpg'
mat = cv.imread(img_path)
df = pd.read_csv('10.txt', header=None, names=['bbox_left', 'bbox_top', 'bbox_width', 'bbox_height', 'confidence', 'object_category'])
df['object_category'] = df['object_category'] + 1
print(df)
image = Image(mat, df, ' ')
image.display(True)