import sys
import argparse
import pandas as pd
import os
import shutil
from utils import env
import matplotlib.pyplot as plt

def format_darknet_annotation(annotation_path, output_folder):
    annotation = pd.read_csv(annotation_path, header=None).values
    for current in annotation:
        open(output_folder + '/{}'.format(annotation_path.split('/')[-1]), 'w+').close()
        image = plt.imread(annotation_path.replace('annotations', 'images').replace('.txt', '.jpg'))
        bbox_left, bbox_top, bbox_width, bbox_height, _, obj_cat, _, _ = current[:8]
        x_center = (bbox_left + bbox_width/2) / image.shape[1]
        y_center = (bbox_top + bbox_height/2) / image.shape[0]
        width = bbox_width / image.shape[1]
        height = bbox_height / image.shape[0]
        if obj_cat in [1, 2]:
            new = ' '.join(['0', str(x_center), str(y_center), str(width), str(height)])
            with open(output_folder + '/{}'.format(annotation_path.split('/')[-1]), 'a+') as out:
                out.write(new + '\n')


def format_tf(annotation_folder, output_file):
    counter = 1
    for annotation_file in os.listdir(annotation_folder):
        print('Processing ... ({:d}%)'.format( int(counter*100/len(os.listdir(annotation_folder)))), end='\r')
        annotation_file_ = annotation_folder + '/' + annotation_file
        img_path = annotation_file_.replace('annotations', 'images').replace('.txt', '.jpg')
        new = img_path + ' '
        for current in pd.read_csv(annotation_file_, header=None).values:
            bbox_left, bbox_top, bbox_width, bbox_height, _1, obj_cat, _2, _3 = current[:8]
            x_min = bbox_left
            x_max = x_min + bbox_width
            y_max = bbox_top
            y_min = y_max + bbox_height
            if obj_cat in [1, 2]:
                new += ','.join([str(x_min), str(y_min), str(x_max), str(y_max), '0']) + ' '
        new = new.strip()
        if output_file != 0:
            with open(output_file, 'a+') as out:
                out.write(new + '\n')
        else:
            pass
        counter += 1
    print('Done!' + ' '*20)


def format_darknet(output_folder):
    if os.path.exists(output_folder):
        delete = ''
        while delete not in ('y', 'n'):
            delete = input('Folder {} already exists. Are you sure you want to replace it ? (y/n): '.format(output_folder))
            if delete == 'y':
                shutil.rmtree(output_folder)
                break
            elif delete == 'n':
                sys.exit(0)
    
    print('Creating folders ...', end='\r')
    # mkdir data/obj
    os.makedirs(output_folder + '/obj', mode=0o777)
    print('Creating configuration files ...', end='\r')
    # echo 'obj' > data/obj.names
    with open(output_folder + '/obj.names', 'w+') as names:
        names.write('person')
    # Writing obj.data
    with open(output_folder + '/obj.data', 'w+') as out:
        out.write('classes = 1\n')
        out.write('train = {}/train.txt\n'.format(output_folder))
        out.write('valid = {}/valid.txt\n'.format(output_folder))
        out.write('names = {}/obj.names\n'.format(output_folder))
        out.write('backup = backup/')
    # Writing train.txt & valid.txt
    with open(output_folder + '/train.txt', 'w+') as out:
        for image in os.listdir(env.DATASET_ + 'train/images'):
            out.write(output_folder + '/obj/' + image + '\n')
    with open(output_folder + '/valid.txt', 'w+') as out:
        for image in os.listdir(env.DATASET_ + 'validation/images'):
            out.write(output_folder +  '/obj/' + image + '\n')
    # cp VisDrone2019/__subset__/images/* data/obj/ & Formatting annotations
    subsets = ['train/', 'validation/']
    for subset in subsets:
        counter = 1
        length = len(os.listdir(env.DATASET_ + subset + 'images'))
        for image in os.listdir(env.DATASET_ + subset + 'images'):
            print('Preparing data ... ({} - {}%)'.format(subset[:-1], int(counter*100/length)) + ' '*30, end='\r')
            shutil.copy(env.DATASET_ + subset + 'images/' + image, output_folder + '/obj')
            format_darknet_annotation(env.DATASET_ + subset + 'annotations/' + image.replace('jpg', 'txt'), output_folder + '/obj')
            counter += 1
    print('Done!' + ' '*50)


def main():
    description = 'This script formats image annotations to fit a certain format. Note that this script is specific to VisDrone2019 dataset.'

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-s', '--subset', help='specify which dataset subset to process (specific to tf format)', choices=['train', 'test', 'validation'])
    parser.add_argument('-o', '--output', help='specify path to the output file/folder depending on the specified format', required=True)
    parser.add_argument('-f', '--format', help='specify desired format (tf or darknet)', required=True, choices=['tf', 'darknet'])
    args = parser.parse_args()

    output = args.output

    if args.format == 'tf':
        if args.subset == None:
            parser.error('--format tf requires --output argument')
        else:
            annotation_folder = env.TRAIN_PATH_ + 'annotations' if args.subset == 'train' else (env.TEST_PATH_ + 'annotations' if args.subset == 'test' else env.VALIDATION_PATH_ + 'annotations')
            format_tf(annotation_folder, output)
    elif args.format == 'darknet':
        # print('Not implemented yet. Please do it you lazy beautiful cunt.')
        format_darknet(output)


if __name__ == '__main__':
    main()
