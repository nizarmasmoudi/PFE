import argparse
import os
from utils.preprocessing import split_annotation, split_image

def main():
    description = 'Split images and their corresponding annotations into 4 equally shaped pieces'
    parser = argparse.ArgumentParser(description = description)
    
    parser.add_argument('-f', '--folder', help = 'Path to parent folder (should contain image and annotation folder)', required = True)
    parser.add_argument('-v', '--overlap', help = 'Splitting overlap', required = True)
    parser.add_argument('-o', '--output', help = 'Output parent folder', required = True)
    args = parser.parse_args()
    
    images = [args.folder + '/images/' + image_ for image_ in os.listdir(args.folder + '/images')]
    
    if not os.path.exists(args.output + '/images'): os.makedirs(args.output + '/images')
    if not os.path.exists(args.output + '/annotations'): os.makedirs(args.output + '/annotations')
    
    for i, img_path in enumerate(images):
        ann_path = img_path.replace('images', 'annotations').replace('.jpg', '.txt')
        split_image(img_path, int(args.overlap), save_output = args.output + '/images')
        split_annotation(ann_path, overlap = int(args.overlap), save_output = args.output + '/annotations')
        print('Splitting ... ({}%)'.format(i*100//len(images)), end = '\r')
    print('Done' + ' '*20)
    
    
if __name__ == '__main__':
    main()