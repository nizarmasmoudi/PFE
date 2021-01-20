import cv2
import pandas as pd
import argparse

def main():
    description = 'Display image with annotated bounding boxes'
    parser = argparse.ArgumentParser(description = description)
    
    parser.add_argument('-i', '--image', help = 'Path to image', required = True)
    args = parser.parse_args()
    
    img = cv2.imread(args.image)
    try:
        annotations = pd.read_csv(args.image.replace('images', 'annotations').replace('.jpg', '.txt'), header = None).values
    except:
        annotations = []
    finally:
        if len(annotations) < 0:
            pass
        else:
            for left, top, width, height, _, obj, _, _ in annotations:
                if obj in [1, 2]: cv2.rectangle(img, (left, top), (left+width, top+height), (0, 0, 255), 1)
        cv2.imshow(args.image.split('/')[-1], img)
        cv2.waitKey(0)
    
if __name__ == '__main__':
    main()