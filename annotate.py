import argparse
import cv2
import os
import pandas as pd
from tqdm import tqdm

def main():
    description = 'Draw bounding boxes on a given set of images'
    parser = argparse.ArgumentParser(description=description)
    
    parser.add_argument('-f', '--folder', help = 'Path to parent folder (should contain image and annotation folder)', required = True)
    parser.add_argument('-o', '--output', help = 'Output parent folder', required = True)
    args = parser.parse_args()
    
    images = [args.folder + '/images/' + image_ for image_ in os.listdir(args.folder + '/images')]
    
    for i in tqdm(range(len(images)), desc='Annotating', unit=' image'):
        image = images[i]
        img = cv2.imread(image)
        try:
            annotations = pd.read_csv(image.replace('images', 'annotations').replace('.jpg', '.txt'), header = None).values
        except:
            annotations = []
        finally:
            if len(annotations) < 0:
                pass
            else:
                for left, top, width, height, _, obj, _, _ in annotations:
                    if obj in [1, 2]: cv2.rectangle(img, (left, top), (left+width, top+height), (0, 0, 255), 1)
            cv2.imwrite(args.output + '/' + image.split('/')[-1], img)

if __name__ == '__main__':
    main()