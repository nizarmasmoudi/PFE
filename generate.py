import argparse
import os
from utils.preprocessing import process_annotations
import shutil

def main():
    description = 'Process annotations to fit a certain format'
    parser = argparse.ArgumentParser(description = description)
    
    parser.add_argument('-d', '--dataset', help = 'Path to dataset folder', required = True)
    parser.add_argument('-o', '--output', help = 'Path to output folder', required = True)
    args = parser.parse_args()
    
    # Creating squeleton
    os.makedirs(args.output + '/obj')
    with open(args.output + '/obj.names', 'w+') as names: names.write('person')
    with open(args.output + '/obj.data', 'w+') as out:
        out.write('classes = 1\n')
        out.write('train = {}/train.txt\n'.format(args.output))
        out.write('valid = {}/valid.txt\n'.format(args.output))
        out.write('names = {}/obj.names\n'.format(args.output))
        out.write('backup = backup/')
    with open(args.output + '/train.txt', 'w+') as out:
        for image in os.listdir(args.dataset + '/train/images'):
            out.write(args.output + '/obj/' + image + '\n')
    with open(args.output + '/valid.txt', 'w+') as out:
        for image in os.listdir(args.dataset + '/validation/images'):
            out.write(args.output +  '/obj/' + image + '\n')
    with open(args.output + '/test.txt', 'w+') as out:
        for image in os.listdir(args.dataset + '/test/images'):
            out.write(args.output +  '/obj/' + image + '\n')
    # Copying images / Processing annotations
    for subset in ['/train', '/validation', '/test']:
        images = [args.dataset + subset + '/images/' + image for image in os.listdir(args.dataset + subset + '/images')]
        for i, img_path in enumerate(images):
            shutil.copy(img_path, args.output + '/obj')
            ann_path = img_path.replace('images', 'annotations').replace('.jpg', '.txt')
            output = process_annotations(ann_path)
            if output == []:
                open(args.output + '/obj/' + ann_path.split('/')[-1], 'w+').close()
                continue
            output = [line + '\n' for line in output]
            with open(args.output + '/obj/' + ann_path.split('/')[-1], 'w+') as out:
                out.writelines(output)
            print('Making dataset ... ({}%)'.format(i*100//len(images)), end='\r')
    print('Done' + ' '*22)
    

if __name__ == '__main__':
    main()