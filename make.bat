:: Splitting images and annotations
ECHO OFF
ECHO Splitting images and annotations (train)
PAUSE
python split_images.py --subset train --output VisDrone2019_/train
ECHO Splitting images and annotations (test)
PAUSE
python split_images.py --subset test --output VisDrone2019_/test
ECHO Splitting images and annotations (validation)
PAUSE
python split_images.py --subset validation --output VisDrone2019_/validation
:: Cleaning dataset
ECHO Cleaning dataset (train)
PAUSE
python clean.py --folder ./VisDrone2019_/train --summary --negatives --large --log log_train.txt
ECHO Cleaning dataset (validation)
PAUSE
python clean.py --folder ./VisDrone2019_/validation --summary --negatives --large --log log_validation.txt
ECHO Cleaning dataset (test)
PAUSE
python clean.py --folder ./VisDrone2019_/test --summary --negatives --large --log log_test.txt
:: Making darknet dataset
ECHO Making darknet dataset (train)
PAUSE
python generate.py --subset train --output ./data --format darknet
ECHO Making darknet dataset (validation)
PAUSE
python generate.py --subset validation --output ./data --format darknet
ECHO Making darknet dataset (test)
PAUSE
python generate.py --subset test --output ./data --format darknet