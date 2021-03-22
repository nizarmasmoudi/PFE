import os
import pandas as pd
import cv2 as cv
from utils import env
from dataset.preparation import Image

if os.path.exists(r'trash\valid.csv'):
    df = pd.read_csv(r'trash\valid.csv')
    for img_path, looked_at, _ in df.values:
        print('Went through {:.2f}%'.format(df['looked at'].sum()*100/len(df)), end='\r')
        # ! Automatic selection
        if looked_at:
            continue
        image = Image(img_path)
        if len(image.read_annotations()) == 0:
            df.loc[df['image'] == img_path, 'keep'] = False
            df.loc[df['image'] == img_path, 'looked at'] = True 
            continue
        
        # ! Manual selection
        cv.imshow('', image.img)
        while True:
            k = cv.waitKey(33)
            if k == 8:
                df.loc[df['image'] == img_path, 'keep'] = False
                df.loc[df['image'] == img_path, 'looked at'] = True
                break
            elif k == 13:
                df.loc[df['image'] == img_path, 'keep'] = True
                df.loc[df['image'] == img_path, 'looked at'] = True 
                break
            elif k == 27:
                break
        if k == 27:
            break
        
    # ! Save checkpoint
    df.to_csv(r'trash\valid.csv', index=False)
    
else:
    df = pd.DataFrame()
    img_paths = [os.path.join(env.VALID_PATH, 'images', img_path) for img_path in os.listdir(os.path.join(env.VALID_PATH, 'images'))]
    df['image'] = img_paths
    df['looked at'] = False
    df['keep'] = True
    df.to_csv(r'trash\valid.csv', index=False)    