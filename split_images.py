import argparse
from utils.preprocessing import split_annotation, split_image
import os
from utils import env

def main():
    description = 'This script splits images along with their annotations into 4 equally shaped images'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-s', '--subset', help='specify which dataset subset to process', choices=['train', 'test', 'validation'], required=True)
    parser.add_argument('-o', '--output', help='specify the output folder of images and annotations. Two sub-folders will be automatically created (images and annotations)', required=True)
    args = parser.parse_args()
    
    images = os.listdir(env.DATASET_ + args.subset + '/images')
    annot_files = os.listdir(env.DATASET_ + args.subset + '/annotations')
    
    if not os.path.exists(args.output + '/images'):
        os.makedirs(args.output + '/images')
    if not os.path.exists(args.output + '/annotations'):
        os.makedirs(args.output + '/annotations')
    for i, image in enumerate(images):
        print('Splitting images ... ({}%)'.format(int(i*100/len(images))), end='\r')
        split_image(env.DATASET_ + args.subset + '/images/' + image, overlap=20, save_output=args.output + '/images')
    for i, annot_file in enumerate(annot_files):
        print('Splitting annotations ... ({}%)'.format(int(i*100/len(annot_files))), end='\r')
        split_annotation(env.DATASET_ + args.subset + '/annotations/' + annot_file, overlap=20, save_output=args.output + '/annotations')
    

if __name__ == '__main__':
    main()