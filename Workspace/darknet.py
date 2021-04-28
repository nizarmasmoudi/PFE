from processing.image import read_image, Image
import dataset.navigation as nav
from tqdm import tqdm
import numpy as np
import cv2 as cv
import argparse
import os

def process(img_paths: list, output: str, n_classes: int, desc: str = 'Processing') -> list:
    img_paths_ = []
    for k in tqdm(range(len(img_paths)), desc = desc, unit = ' image'):
        image = read_image(img_paths[k])
        image.patch()
        image.pad()
        if image.width == image.height == 736:
            image.parse_annotations(inplace = True, n_classes = n_classes)
            if len(image.annotation) == 0:
                if np.random.random() < .3:
                    cv.imwrite(os.path.join(output, 'data', image.title), image.mat)
                    open(os.path.join(output, 'data', image.title.replace('.jpg', '.txt')), 'w+').close()
            else:
                cv.imwrite(os.path.join(output, 'obj', image.title), image.mat)
                image.annotation.to_csv(os.path.join(output, 'obj', image.title.replace('.jpg', '.txt')), header=False, index=False, sep=' ')
            img_paths_.append(os.path.join(output, 'obj', image.title))
            continue
        elif image.width == 736 and image.height > 736:
            crop_locs = ['upper left', 'lower left']
        elif image.height == 736 and image.width > 736:
            crop_locs = ['upper left', 'upper right']
        else:
            crop_locs = ['upper left', 'lower left', 'upper right', 'lower right']
        crops = [image.crop(loc) for loc in crop_locs]
        for crop in crops:
            # crop.filter()
            crop.parse_annotations(inplace = True, n_classes = n_classes)
            if len(crop.annotation) == 0:
                if np.random.random() < .3:
                    cv.imwrite(os.path.join(output, 'obj', crop.title), crop.mat)
                    open(os.path.join(output, 'obj', crop.title.replace('.jpg', '.txt')), 'w+').close()
                    img_paths_.append(os.path.join(output, 'obj', crop.title))
            else:
                cv.imwrite(os.path.join(output, 'obj', crop.title), crop.mat)
                crop.annotation.to_csv(os.path.join(output, 'obj', crop.title.replace('.jpg', '.txt')), header=False, index=False, sep=' ')
                img_paths_.append(os.path.join(output, 'obj', crop.title))
    return img_paths_

def main():
    parser = argparse.ArgumentParser(description = 'Generate training dataset for Darknet framework.')
    parser.add_argument('--source', '-s', type = str, help = 'Source folder of original dataset (Default: [./dataset/navigation.py]', default = nav.DATASET_PATH)
    parser.add_argument('--output', '-o', type = str, help = 'Output path of processed data', required = True)
    parser.add_argument('--classes', '-c', type = int, help = 'Number of classes (Default: 1)', default = 1)
    parser.add_argument('--subsets', '-u', type = str, nargs = '+', help = 'Subsets to process.', choices = ['train', 'valid', 'test'], required = True)
    args = parser.parse_args()
    
    print(
        f'''
        Processing data:
            Source = {args.source},
            Output = {args.output},
            Number of classes = {args.classes},
            Subsets = {' '.join(args.subsets)},
        '''
    )
    confirm = ''
    while confirm not in ['y', 'n']:
        confirm = input('Start processing ? [y|n]: ')
        if confirm == 'n':
            exit()
        elif confirm == 'y':
            break
    
    if 'train' in args.subsets:
        train_imgs = [os.path.join(args.source, 'train', 'images', img_path) for img_path in os.listdir(os.path.join(args.source, 'train', 'images'))]
    if 'valid' in args.subsets:
        valid_imgs = [os.path.join(args.source, 'validation', 'images', img_path) for img_path in os.listdir(os.path.join(args.source, 'validation', 'images'))]
    if 'test' in args.subsets:
        test_imgs = [os.path.join(args.source, 'test', 'images', img_path) for img_path in os.listdir(os.path.join(args.source, 'test', 'images'))]
    
    os.makedirs(args.output, exist_ok = True)
    os.makedirs(os.path.join(args.output, 'obj'), exist_ok = True)
    with open(os.path.join(args.output, 'obj.data'), 'w+') as data:
        classes = args.classes
        train = os.path.join(args.output, 'data', 'train.txt')
        valid = os.path.join(args.output, 'data', 'valid.txt')
        names = os.path.join(args.output, 'data', 'names.txt')
        data.writelines([
            f'classes = {classes}\n',
            f'valid = {train}\n',
            f'valid = {valid}\n',
            f'names = {names}\n',
            f'backup = backup/'
        ])
    with open(os.path.join(args.output, 'obj.names'), 'w+') as names:
        names.write('Person') if classes == 1 else names.writelines(['Pedestrian', 'People'])
    
    if 'train' in args.subsets:
        train_imgs_ = process(train_imgs, output = args.output, n_classes = args.classes, desc='Processing training images')
        with open(os.path.join(args.output, 'train.txt'), 'w+') as train:
            train.writelines([path + '\n' for path in train_imgs_])
        
    if 'valid' in args.subsets:
        valid_imgs_ = process(valid_imgs, output = args.output, n_classes = args.classes, desc='Processing validation images')
        with open(os.path.join(args.output, 'valid.txt'), 'w+') as valid:
            valid.writelines([path + '\n' for path in valid_imgs_])
    
    if 'test' in args.subsets:
        test_imgs_ = process(test_imgs, output = args.output, n_classes = args.classes, desc='Processing test images')
        with open(os.path.join(args.output, 'test.txt'), 'w+') as test:
            test.writelines([path + '\n' for path in test_imgs_])
    
if __name__ == '__main__':
    main()