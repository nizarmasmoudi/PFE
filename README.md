<h1 align='center'>Real-time Video Traffic Analysis for Crowd Control in post-COVID-19 Public Events</h1>

## Dataset
Currently I am working with [VisDrone2019](https://github.com/VisDrone/VisDrone-Dataset) dataset. Only the first task in the repository will be considered (Object Detection in Images).

Here's a quick summary of the dataset :
|              |     Train    | Validation |     Test     |
|:------------:|:------------:|:----------:|:------------:|
|   **Count**  |   *6 471*    |    *548*   |   *1 580*    |
| **Max width**|   *2 000*    |   *1 920*  |   *1 920*    |
|**Max height**|   *1 500*    |   *1080*   |   *1 080*    |

Annotation are represented as follows :
```
    <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
```
Since people are the only subject of interest, only labels `1` and `2` will be considered (pedestrian, people)

For scripts to work, it is highly advised to strcuture data folder in such manner:
```
VisDrone2019
├───test
│   ├───annotations
│   └───images
├───train
│   ├───annotations
│   └───images
└───validation
    ├───annotations
    └───images
```

*Note : The dataset folder isn't visible in this repository for size issues. Please click on the link above to download it*

## Scripts
- `generate.py` (generates dataset according to a certain format)

    ```
    optional arguments:
        -h, --help
                show this help message and exit
        -s {train,test,validation}, --subset {train,test,validation}
                specify which dataset subset to process (specific to tf format)
        -o OUTPUT, --output OUTPUT
                specify path to the output file/folder depending on the specified format
        -f {tf,darknet}, --format {tf,darknet}
                specify desired format (tf or darknet)
    ```
- `split_images.py` (splits images along with their annotations to reduce network size when training)

    ```
    optional arguments:
        -h, --help
                show this help message and exit
        -s {train,test,validation}, --subset {train,test,validation}
                specify which dataset subset to process
        -o OUTPUT, --output OUTPUT
                specify the output folder of images and annotations. Two sub-folders will be automatically created (images and annotations)
    ```
- `clean_negatives.py` (removes negative samples in case they are causing a problem)

    ```
    optional arguments:
        -h, --help
                show this help message and exit
        -f FOLDER, --folder FOLDER
                main folder containing image folder and annotation folder
        -s, --summary
                Print summary at the end of the script
        -l LOG, --log LOG
                Log deleted images in a log file
    ```

## Object Detection Algorithm
I am working on training a [YOLOv4](https://arxiv.org/abs/2004.10934) model using [Darknet](https://github.com/AlexeyAB/darknet) framework.

This framework requires images to be annotated as follows:
```
    <object_class> <x_center> <y_center> <width> <height>
```

Dataset must be organised in such manner:
```
data
│   obj.data
│   obj.names
│   train.txt
│   valid.txt
│
└───obj
```

*Note : All values are relative to the width and height of the image (between 0 and 1)*

Another alternative is to train using [Tensorflow](https://github.com/SoloSynth1/tensorflow-yolov4). It's not advised. However, it might come handy later when deploying the model (Detect people using tensorflow implemented YOLOv4 and Darknet generated weights file).

In case of need, here's the annotation format required:
```
    <img_path> <x_min>,<y_min>,<x_max>,<y_max>,<object_category> <x_min>,<y_min>,<x_max>,<y_max>,<object_category> ...
```