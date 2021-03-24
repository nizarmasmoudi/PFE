from numpy.lib.npyio import save
import pandas as pd
import numpy as np
import cv2 as cv
import os

class Image:
    def __init__(self, img_path, raw=1):
        self.img_path = img_path
        if not os.path.exists(self.img_path):
            raise FileNotFoundError(self.img_path)
        self.ann_path = self.img_path.replace('images', 'annotations').replace('.jpg', '.txt')
        if not os.path.exists(self.ann_path):
            raise FileNotFoundError(self.ann_path)
        self.title = os.path.basename(self.img_path)
        self.mat = cv.imread(self.img_path)
        self.height = self.mat.shape[0]
        self.width = self.mat.shape[1]
        self.annotations = (
            pd.read_csv(
                self.ann_path,
                header=None,
                usecols=[0, 1, 2, 3, 5 if raw else 4],
                names=['left', 'top', 'width', 'height', 'object_category']
            )[lambda df: df['object_category'].isin([0, 1, 2])]
            .dropna(axis = 'columns')
        )
    
    def display(self, bbox=True):
        '''Display image with OpenCV.

        Args:
            bbox (bool, optional): Display bounding boxes and ignored annotations. Defaults to True.
        '''
        if bbox:
            mat_ = self.mat.copy()
            for left, top, width, height, obj in self.annotations.values:
                if obj in [1, 2]:
                    cv.rectangle(mat_, (left, top), (left+width, top+height), (0, 255, 0), 1)
                    # cv.putText(mat_, str(height), (left, top - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                elif obj == 0:
                    cv.rectangle(mat_, (left, top), (left+width, top+height), (255, 255, 255), 1)
                    for wp in range(left, left + width, 5):
                        cv.line(mat_, (wp, top), (wp, top + height), (255, 255, 255), 1)
                    for vp in range(top, top + height, 5):
                        cv.line(mat_, (left, vp), (left + width, vp), (255, 255, 255), 1)
            cv.imshow(self.title, mat_)
            cv.waitKey(0)
        else:
            cv.imshow(self.title, self.mat)
            cv.waitKey(0)
            
    def patch(self):
        '''Fill ignored regions with white noise.
        '''
        regions = self.annotations.loc[self.annotations['object_category'] == 0, ['left', 'top', 'width', 'height']]
        if len(regions) > 0:
            mat_ = np.copy(self.mat)
            for left, top, width, height in regions.values:
                mat_[top:top+height, left:left+width, ...] = np.random.normal(0, 1, size = (height, width, 3))
            self.mat = mat_
            
    def pad(self, target):
        '''Pad image with zeros to fit target shape. Padding doesn't affect annotations.

        Args:
            target (tuple): Target shape.
        '''
        height, width = target
        self.mat = np.pad(self.mat, ((0, max(height - self.height, 0)), (0, max(width - self.width, 0)), (0, 0)), 'constant', constant_values=0)
        
    def split(self, save_images=None, save_annotations=None, target=(736, 736)):
        '''Split image into 4 equal splits. Annotations will be split as well with dimensions taken into consideration.

        Args:
            save_images (string, optional): Path to parent directory where split images will be saved. Defaults to None. If not specified split images will be returned.
            save_annotations (string, optional): Path to parent directory where split annotations will be saved. Defaults to None. If not specified split annotations will be returned.
            target (tuple, optional): Target shape of split images. Defaults to (736, 736).

        Returns:
            tuple: Annotations and matrices of split images
        '''
        images = [
            self.mat[:self.height//2, :self.width//2, :], # Top left 
            self.mat[:self.height//2, self.width//2:, :], # Top right
            self.mat[self.height//2:, :self.width//2, :], # Bottom left
            self.mat[self.height//2:, self.width//2:, :], # Bottom right
        ]
        #! This is the only changement
        images = [np.pad(image, ((0, max(target[0] - self.height//2, 0)), (0, max(target[1] - self.width//2, 0)), (0, 0)), 'constant', constant_values=0) for image in images]
        
        map_ = {'top_left': [], 'top_right': [], 'bottom_left': [], 'bottom_right': []}
        objects = self.annotations.loc[self.annotations['object_category'] != 0]
        for left, top, width, height, obj_cat in objects.values:
            y_center, x_center = top + height//2, left + width//2
            if y_center - height//4 < self.height//2 and x_center - width//4 < self.width//2:
                map_['top_left'].append([left, top, min(width, self.width//2 - left), min(height, self.height//2 - top), obj_cat])
            if y_center - height//4 < self.height//2 and x_center + width//4 >= self.width//2:
                map_['top_right'].append([max(left, self.width//2) - (self.width//2), top, width, min(height, self.height//2 - top), obj_cat])
            if y_center + height//4 >= self.height//2 and x_center - width//4 < self.width//2:
                map_['bottom_left'].append([left, max(top, self.height//2) - (self.height//2), min(width, self.width//2 - left), height - max(0, self.height//2 - top), obj_cat])
            if y_center + height//4 >= self.height//2 and x_center + width//4 >= self.width//2:
                map_['bottom_right'].append([max(left, self.width//2) - (self.width//2), max(top, self.height//2) - (self.height//2), width, height - max(0, self.height//2 - top), obj_cat])
            
        map_['top_left'] = pd.DataFrame(map_['top_left'])
        map_['top_right'] = pd.DataFrame(map_['top_right'])
        map_['bottom_left'] = pd.DataFrame(map_['bottom_left'])
        map_['bottom_right'] = pd.DataFrame(map_['bottom_right'])
        
        if save_images:
            for i, image in enumerate(images):
                cv.imwrite(os.path.join(save_images, self.title.replace('.', '_{}.'.format(i + 1))), image)
        else:
            return images
        if save_annotations:
            for i, loc in enumerate(map_.keys()):
                map_[loc].to_csv(os.path.join(save_annotations, self.title.replace('.jpg', '_{}.txt'.format(i + 1))), header=False, index=False)
        else:
            return map_, images
      
    def parse_annotations(self, save_output=None):
        annotations = self.annotations.copy().loc[self.annotations['object_category'] > 0, :]
        annotations['x_center'] = (annotations['left'] + annotations['width']/2)/self.width
        annotations['y_center'] = (annotations['top'] + annotations['height']/2)/self.height
        annotations['width'] = annotations['width']/self.width
        annotations['height'] = annotations['height']/self.height
        annotations['object_category'] = 0
        annotations = annotations.loc[:, ['object_category', 'x_center', 'y_center', 'width', 'height']]
        if save_output:
            annotations.to_csv(os.path.join(save_output, os.path.basename(self.ann_path)), sep=' ', header=False, index=False)
        else:
            return annotations
          
      
    # def parse_annotations(self, save_output=None):
    #     if len(self.annotations) > 0:
    #         parsed = []
    #         for left, top, width, height, obj_cat in self.annotations.values:
    #             x_center = (left + width/2) / self.width
    #             y_center = (top + height/2) / self.height
    #             width = width/self.width
    #             height = height/self.height
    #             parsed.append('0 {} {} {} {}\n'.format(x_center, y_center, width, height))
    #         if save_output:
    #             with open(os.path.join(save_output, self.title.replace('.jpg', '.txt')), 'w+') as out:
    #                 out.writelines(parsed)
    #         else:
    #             return parsed
    #     else:
    #         if save_output:
    #             open(os.path.join(save_output, self.title.replace('.jpg', '.txt')), 'w+').close()
    #         else:
    #             return []