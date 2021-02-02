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
- `generate.py` (generates dataset according to darknet format)

    ```
    optional arguments:
        -h, --help            show this help message and exit
        -d DATASET, --dataset DATASET
                                Path to dataset folder
        -o OUTPUT, --output OUTPUT
                                Path to output folder
- `split.py` (splits images along with their annotations to reduce network size when training)

    ```
    optional arguments:
        -h, --help            show this help message and exit
        -f FOLDER, --folder FOLDER
                                Path to parent folder (should contain image and annotation folder)
        -v OVERLAP, --overlap OVERLAP
                                Splitting overlap
        -o OUTPUT, --output OUTPUT
                                Output parent folder
    ```
- `clean.py` (removes negative samples and/or large images)

    ```
    optional arguments:
        -h, --help            show this help message and exit
        -f FOLDER, --folder FOLDER
                                Path to parent folder (should contain image and annotation folder)
        -n, --negatives       Delete negative samples
        -l, --large           Delete large images
        -t THRESHOLD, --threshold THRESHOLD
                                Threshold of width/height for deletion
        -g, --log             Log deleted files
    ```
- `display.py` (displays image with bounding boxes)

    ```
    optional arguments:
        -h, --help            show this help message and exit
        -i IMAGE, --image IMAGE
                                Path to image
    ```
- `annotate.py` (re-creates a set of images adding bounding boxes)

    ```
    optional arguments:
        -h, --help            show this help message and exit
        -f FOLDER, --folder FOLDER
                                Path to parent folder (should contain image and annotation folder)
        -o OUTPUT, --output OUTPUT
                                Output parent folder
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