import argparse
import os

from numpy.lib.npyio import save
from utils import env
from dataset.image import Image
from tqdm import tqdm

def fill_ignored_regions(image):
    annotations = image.read_annotations(subclasses = [0])
    if len(annotations) > 0:
        for left, top, width, height, _, _, _, _ in annotations.values:
            image.add_noise([left, top, width, height])
    return image

def main():
    description = 'Generate input for Darknet YOLO'
    parser = argparse.ArgumentParser(description=description)
    
    parser.add_argument('-o', '--output', help = 'Path to output folder', required = True)
    parser.add_argument('-c', '--clean', help = 'Delete large images and negative samples', action = 'store_true')
    args = parser.parse_args()

    ## Configuration files
    with open(os.path.join(args.output, 'obj.names'), 'w+') as names: names.write('person')
    with open(os.path.join(args.output, 'obj.data'), 'w+') as data: 
        data.writelines([
            'classes = 1\n',
            'train = {}\n'.format(os.path.join(args.output), 'train.txt'),
            'valid = {}\n'.format(os.path.join(args.output), 'valid.txt'),
            'names = {}\n'.format(os.path.join(args.output), 'obj.names'),
            'backup = {}\n'.format(os.path.join(args.output), 'backup'),
        ])
    with open(os.path.join(args.output, 'train.txt'), 'w+') as train:
        img_paths = [os.path.join(args.output, img_title) + '\n' for img_title in os.listdir(os.path.join(env.TRAIN_PATH, 'images'))]
        train.writelines(img_paths)
    with open(os.path.join(args.output, 'valid.txt'), 'w+') as train:
        img_paths = [os.path.join(args.output, img_title) + '\n' for img_title in os.listdir(os.path.join(env.VALID_PATH, 'images'))]
        train.writelines(img_paths)
    
    ## Data folder
    os.makedirs(os.path.join(args.output, 'obj'), exist_ok=True)
    for subset_path in [env.TRAIN_PATH, env.VALID_PATH]:
        img_paths = [os.path.join(subset_path, 'images', img_title) for img_title in os.listdir(os.path.join(subset_path, 'images'))]
        for i in tqdm(range(len(img_paths)), unit='image'):
            image = Image(img_paths[i])
            image = fill_ignored_regions(image)
            image.split(overlap=20, save_output={'images': os.path.join(args.output, 'obj'), 'annotations': os.path.join(args.output, 'obj')})
            for i in range(1, 5):
                image_ = Image(os.path.join(args.output, 'obj', image.id + '_{}.jpg'.format(i)))
                image_.parse_annotations(save_output=os.path.join(args.output, 'obj'))
    
    ## Cleaning
    if args.clean:
        img_paths = [os.path.join(args.output, 'obj', img_title) for img_title in os.listdir(os.path.join(args.output, 'obj')) if img_title.endswith('.jpg')]
        for i in tqdm(range(len(img_paths)), unit='image'):
            image = Image(img_paths[i])
            if os.stat(image.ann_path).st_size == 0:
                os.remove(image.img_path)
                os.remove(image.ann_path)
            elif image.width > 736 or image.height > 736:
                os.remove(image.img_path)
                os.remove(image.ann_path)
                

if __name__ == '__main__':
    main()