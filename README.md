<h1 align="center">Crowd Counting and Monitoring</h1>

## Tasks
- [x] Format annotations
    - [x] Format annotations to fit Tensorflow implementation requirements
    ```
    python format --subset [train, test, validation] --output path/to/output/file.txt --format tf
    ```
    - [x] Format annotations to fit Darknet framework requirements (*this one takes a while to run)
    ```
    python format --output path/to/data/folder --format darknet
    ```
- [ ] Train a YOLOv4 model on custom dataset (using Darknet or Tensorflow)
- [ ] Crowd density entimation (K-Means, DBScan, ...)
- [ ] ...

## Datasets
- ### [VisDrone](https://github.com/VisDrone/VisDrone-Dataset)

We will consider the first task in the repository (Object Detection in Images).

Note that Crowd counting task isn't compatible for training a YOLO algorithm.

|              |     Train    | Validation |     Test     |
|:------------:|:------------:|:----------:|:------------:|
|   **Count**  |   *6 471*    |    *548*   |   *1 580*    |
| **Max width**|   *2 000*    |   *1 920*  |   *1 920*    |
|**Max height**|   *1 500*    |   *1080*   |   *1 080*    |

```
<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
```

There are multiple object categories. However, we will only consider labels [1, 2] = [Pedestrian, People]

Note: VisDrone2019 isn't visible in this github repository for memory usage reasons. However, here's the structure of the folder:
```
|- VisDrone2019
|    |- train
|    |    |- images
|    |    |    |- 0000002_00005_d_0000014.jpg
|    |    |    |- ...
|    |    |- annotations
|    |    |    |- 0000002_00005_d_0000014.txt
|    |    |    |- ...
|    |- validation
|    |    |- images
|    |    |    |- 0000001_02999_d_0000005.jpg
|    |    |    |- ...
|    |    |- annotations
|    |    |    |- 0000001_02999_d_0000005.txt
|    |    |    |- ...
|    |- test
|    |    |- images
|    |    |    |- 0000006_00159_d_0000001.jpg
|    |    |    |- ...
|    |    |- annotations
|    |    |    |- 0000006_00159_d_0000001.txt
|    |    |    |- ...
```

## YOLO Model
- ### [YOLOv4 using Tensorflow](https://github.com/SoloSynth1/tensorflow-yolov4)

This repository requires images to be annotated as follows:

```
<img_path> <x_min>,<y_min>,<x_max>,<y_max>,<object_category> <x_min>,<y_min>,<x_max>,<y_max>,<object_category> ...
```

- ### [Darknet Framework](https://github.com/AlexeyAB/darknet)

[yolov4.weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights) is a file containing pre-trained weights for transfer learning.

This framework requires images to be annotated as follows:

```
<object_class> <x_center> <y_center> <width> <height>
```

All values are relative the width and height of the image.