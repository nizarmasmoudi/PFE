import pandas as pd
import numpy as np
import cv2 as cv
import os

def read_image(img_path: str) -> 'Image':
    ann_path = img_path.replace('images', 'annotations').replace('.jpg', '.txt')
    title = os.path.basename(img_path)
    mat = cv.imread(img_path)    
    annotation = pd.read_csv(ann_path, header=None, names=['bbox_left', 'bbox_top', 'bbox_width', 'bbox_height', 'score', 'object_category', 'truncation', 'occlusion'])
    annotation = annotation.loc[annotation['object_category'].isin([0, 1, 2])]
    return Image(mat, annotation, title)

class Image():
    def __init__(self, mat: np.ndarray, annotation: pd.DataFrame, title: str = '') -> None:
        self.mat = mat
        self.annotation = annotation
        self.height = mat.shape[0]
        self.width = mat.shape[1]
        self.title = title
    def display(self, bbox: bool = False) -> None:
        if bbox:
            mat_ = self.mat.copy()
            for left, top, width, height, obj in self.annotation.loc[:, ['bbox_left', 'bbox_top', 'bbox_width', 'bbox_height', 'object_category']].values:
                left, top, width, height, obj = int(left), int(top), int(width), int(height), int(obj)
                if obj == 1:
                    cv.rectangle(mat_, (left, top), (left+width, top+height), (0, 255, 0), 1)
                elif obj == 2:
                    cv.rectangle(mat_, (left, top), (left+width, top+height), (0, 0, 255), 1)
                elif obj == 0:
                    cv.rectangle(mat_, (left, top), (left+width, top+height), (255, 255, 255), 1)
                    for wp in range(left, left + width, 5):
                        cv.line(mat_, (wp, top), (wp, top + height), (255, 255, 255), 1)
                    for vp in range(top, top + height, 5):
                        cv.line(mat_, (left, vp), (left + width, vp), (255, 255, 255), 1)
            cv.imshow(self.title, mat_)
            cv.waitKey(0)
            cv.destroyAllWindows()
        else:
            cv.imshow(self.title, self.mat)
            cv.waitKey(0)
            cv.destroyAllWindows()
    def patch(self) -> None:
        regions = self.annotation.loc[self.annotation['object_category'] == 0, ['bbox_left', 'bbox_top', 'bbox_width', 'bbox_height']]
        if len(regions) > 0:
            mat_ = np.copy(self.mat)
            for left, top, width, height in regions.values:
                mat_[top:top + height, left:left + width, ...] = np.random.normal(0, 1, size = (height, width, 3))
            self.mat = mat_
    def save(self, output_path: str, ann_sep: str = ' ') -> None:
        cv.imwrite(os.path.join(output_path, 'images', self.title), self.mat)
        self.annotation.to_csv(os.path.join(output_path, 'annotations', self.title.replace('.jpg', '.txt')), header=False, index=False, sep=ann_sep)
    def pad(self, target: tuple = (736, 736)) -> None:
        pad_height, pad_width = target
        self.mat = np.pad(self.mat, ((0, max(pad_height - self.height, 0)), (0, max(pad_width - self.width, 0)), (0, 0)), 'constant', constant_values=0)
        self.height, self.width = self.mat.shape[0], self.mat.shape[1]
    def crop(self, loc: str, size: tuple = (736, 736)) -> 'Image':
        locs = ['upper left', 'lower left', 'upper right', 'lower right']
        if loc not in locs:
            raise Exception(f'loc argument should be one of {locs}')
        
        crop_height, crop_width = size
        if crop_height == self.height and crop_width == self.width:
            print('Image already fits size', size)
            return
        
        else:
            if loc == 'upper left':
                sub_mat = self.mat[:crop_height, :crop_width, :].copy()
                sub_annotation = self.annotation.loc[(
                    ((self.annotation['bbox_top'] + .5*self.annotation['bbox_height']) < crop_height) & ((self.annotation['bbox_left'] + .5*self.annotation['bbox_width']) < crop_width)
                )].copy()
            elif loc == 'upper right':
                sub_mat = self.mat[:crop_height, self.width - crop_width:, :].copy()
                sub_annotation = self.annotation.loc[(
                    ((self.annotation['bbox_top'] + .5*self.annotation['bbox_height']) < crop_height) & ((self.annotation['bbox_left'] + .5*self.annotation['bbox_width']) > self.width - crop_width)
                )].copy()
                sub_annotation.loc[:, 'bbox_left'] = sub_annotation.loc[:, 'bbox_left'] - (self.width - crop_width)
            elif loc == 'lower left':
                sub_mat = self.mat[self.height - crop_height:, :crop_width, :].copy()
                sub_annotation = self.annotation.loc[(
                    ((self.annotation['bbox_top'] + .5*self.annotation['bbox_height']) > self.height - crop_height) & ((self.annotation['bbox_left'] + .5*self.annotation['bbox_width']) < crop_width)
                )].copy()
                sub_annotation.loc[:, 'bbox_top'] = sub_annotation.loc[:, 'bbox_top'] - (self.height - crop_height)
            elif loc == 'lower right':
                sub_mat = self.mat[self.height - crop_height:, self.width - crop_width:, :].copy()
                sub_annotation = self.annotation.loc[(
                    ((self.annotation['bbox_top'] + .5*self.annotation['bbox_height']) > self.height - crop_height) & ((self.annotation['bbox_left'] + .5*self.annotation['bbox_width']) > self.width - crop_width)
                )].copy()
                sub_annotation.loc[:, 'bbox_top'] = sub_annotation.loc[:, 'bbox_top'] - (self.height - crop_height)
                sub_annotation.loc[:, 'bbox_left'] = sub_annotation.loc[:, 'bbox_left'] - (self.width - crop_width)

            sub_annotation = sub_annotation.reset_index(drop=True)
            if len(sub_annotation) > 0:
                for i in range(len(sub_annotation)):
                    left, top, width, height = sub_annotation.loc[i, ['bbox_left', 'bbox_top', 'bbox_width', 'bbox_height']]
                    if left < 0:
                        width -= abs(left)
                        left = 0
                    if top < 0:
                        height -= abs(top)
                        top = 0
                    sub_annotation.loc[i, ['bbox_left', 'bbox_top', 'bbox_width', 'bbox_height']] = [left, top, width, height]
            
            return Image(sub_mat, sub_annotation, self.title.replace('.jpg', '_{}.jpg'.format(['upper left', 'lower left', 'upper right', 'lower right'].index(loc))))
    def parse_annotations(self, inplace: bool = False, n_classes: int = 1) -> pd.DataFrame:
        parsed = pd.DataFrame(columns=['object_class', 'x_center', 'y_center', 'width', 'height'])
        parsed['object_class'] = self.annotation['object_category']
        parsed['x_center'] = (self.annotation['bbox_left'] + self.annotation['bbox_width']/2)/self.width
        parsed['y_center'] = (self.annotation['bbox_top'] + self.annotation['bbox_height']/2)/self.height
        parsed['width'] = self.annotation['bbox_width']/self.width
        parsed['height'] = self.annotation['bbox_height']/self.height
        parsed = parsed[parsed['object_class'].isin([1, 2])]
        
        if n_classes == 2:
            parsed['object_class'] -= 1
        else:
            parsed['object_class'] = 0
        if inplace:
            self.annotation = parsed[['object_class', 'x_center', 'y_center', 'width', 'height']]
        else:
            return parsed[['object_class', 'x_center', 'y_center', 'width', 'height']]