import matplotlib.pyplot as plt
import os
import argparse
from tqdm import tqdm

def main():
    description = 'Delete large images and/or negative samples'
    parser = argparse.ArgumentParser(description = description)
    
    parser.add_argument('-f', '--folder', help = 'Path to parent folder (should contain image and annotation folder)', required = True)
    parser.add_argument('-n', '--negatives', help = 'Delete negative samples', action = 'store_true')    
    parser.add_argument('-l', '--large', help = 'Delete large images', action = 'store_true')
    parser.add_argument('-t', '--threshold', help = 'Threshold of width/height for deletion', required = True)
    parser.add_argument('-g', '--log', help = 'Log deleted files', action = 'store_true')
    args = parser.parse_args()
    
    if args.negatives:
        images = [args.folder + '/images/' + image_ for image_ in os.listdir(args.folder + '/images')]
        negatives = []
        for i in tqdm(range(len(images)), desc='Scanning negative samples', unit=' file'):
            image = images[i]
            annotations = image.replace('images', 'annotations').replace('.jpg', '.txt')
            if os.stat(annotations).st_size == 0:
                os.remove(image)
                os.remove(annotations)
                negatives.append(image)
        if args.log:
            with open('negatives_log.txt', 'w+') as log:
                log.writelines([negative + '\n' for negative in negatives])
        
    if args.large:
        images = [args.folder + '/images/' + image_ for image_ in os.listdir(args.folder + '/images')]
        large = []
        for i in tqdm(range(len(images)), desc='Scanning large images', unit=' image'):
            image = images[i]
            img = plt.imread(image)
            annotations = image.replace('images', 'annotations').replace('.jpg', '.txt')
            if img.shape[0] > int(args.threshold) or img.shape[1] > int(args.threshold):
                os.remove(image)
                os.remove(annotations)
                large.append(image)
        if args.log:
            with open('large_log.txt', 'w+') as log:
                log.writelines([l + '\n' for l in large])
    
if __name__ == '__main__':
    main()