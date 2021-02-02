import argparse
from utils.preprocessing import fill_ignored_regions
import os
from tqdm import tqdm

def main():
    description = 'Fill ignored regions with patches'
    parser = argparse.ArgumentParser(description = description)

    parser.add_argument('-f', '--folder', help = 'Path to parent folder (should contain image and annotation folder)', required = True)
    parser.add_argument('-o', '--output', help = 'Path output to parent folder')
    parser.add_argument('-i', '--inplace', help = 'overwrite saved images', action = 'store_true')
    
    args = parser.parse_args()
    
    images = [args.folder + '/images/' + image_ for image_ in os.listdir(args.folder + '/images')]
    for i in tqdm(range(len(images)), desc='Filling', unit=' image'):
        img_path = images[i]
        fill_ignored_regions(img_path, inplace=args.inplace, save_output=args.output)

if __name__ == '__main__':
    main()