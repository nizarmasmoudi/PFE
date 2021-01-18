import argparse
import os
import matplotlib.pyplot as plt

def main():
    description = 'This script cleans dataset from negative samples. Careful, this scripts deletes files and doesn\'t create an alternative folder'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-f', '--folder', help='main folder containing image folder and annotation folder', required=True)
    parser.add_argument('-s', '--summary', help='Print summary at the end of the script', action='store_true')
    parser.add_argument('-n', '--negatives', help='Delete negative samples', action='store_true')
    parser.add_argument('-w', '--large', help='Delete large samples', action='store_true')
    parser.add_argument('-l', '--log', help='Log deleted images in a log file')
    args = parser.parse_args()
    
    annot_files = os.listdir(args.folder + '/annotations')
    annot_files = [args.folder + '/annotations/' + file_ for file_ in annot_files]
    if args.negatives:
        empty_files = []
        for annot_file in annot_files:
            if os.stat(annot_file).st_size == 0:
                os.remove(annot_file)
                os.remove(annot_file.replace('annotations', 'images').replace('txt', 'jpg'))
                empty_files.append(annot_file.replace('annotations', 'images').replace('txt', 'jpg'))
    if args.large:
        large_files = []
        for image in [file_.replace('annotations', 'images').replace('.txt', '.jpg') for file_ in annot_files]:
            try:
                img = plt.imread(image)
            except:
                continue
            if img.shape in [(1500, 2000, 3), (1080, 1920, 3)]:
                os.remove(image)
                os.remove(image.replace('images', 'annotations').replace('.jpg', '.txt'))
                large_files.append(image)
    if args.summary:
        print('{} empty annotation files were found and deleted'.format(len(empty_files)))
        print('{} large images were found and deleted'.format(len(large_files)))
    if args.log:
        if len(empty_files) != 0:
            with open(args.log, 'w+') as log:
                log.writelines([file_.split('/')[-1] + '\n' for file_ in empty_files])
                log.writelines([file_.split('/')[-1] + '\n' for file_ in large_files])
        else:
            print('No files deleted. Log file will not be created')

if __name__ == '__main__':
    main() 