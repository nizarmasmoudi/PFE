import pandas as pd
import numpy as np
import cv2
import os

class Image:
    def __init__(self, img_path):
        if os.path.exists(img_path):
            self.img_path = img_path
            self.id = img_path.split('\\')[-1].split('.')[0]
            self.img = cv2.imread(img_path)
            self.height = self.img.shape[0]
            self.width = self.img.shape[1]
            self.ann_path = img_path.replace('images', 'annotations').replace('.jpg', '.txt')
        else:
            raise FileNotFoundError('Image Not Found!\nCheck input path: {}'.format(img_path))
        
    def show(self, draw_bbox = False):
        if draw_bbox:
            annotations = self.read_annotations(subclasses=[1, 2, 0])
            for left, top, width, height, _, obj, _, _ in annotations.values:
                if obj in [1, 2]:
                    cv2.rectangle(self.img, (left, top), (left+width, top+height), (0, 0, 255), 1)
                elif obj == 0:
                    patch = np.ones((height, width, 3), dtype = np.uint8)*255 - 35
                    patch = cv2.addWeighted(self.img[top:top+height, left:left+width], 0.5, patch, 0.5, 1.0)
                    self.img[top:top+height, left:left+width] = patch
                
        cv2.imshow(self.id, self.img)
        cv2.waitKey(0)
        
    def read_annotations(self, subclasses=[1, 2]):
        if os.path.exists(self.ann_path):
            df = pd.read_csv(self.ann_path, header = None, names=['left', 'top', 'width', 'height', 'score', 'object_category', 'truncation', 'occlusion', '_'])
            df.dropna(axis = 'columns', inplace=True)
            df = df[df['object_category'].isin(subclasses)]
            return df
        else:
            raise FileNotFoundError('Annotation File Not Found!')
        
    def split(self, overlap=0, save_output={'images': None, 'annotations': None}):
        ## Splitting image
        pieces = [
            self.img[:self.height//2 + overlap, :self.width//2 + overlap, :], # Top left 
            self.img[self.height//2 - overlap:, :self.width//2 + overlap, :], # Bottom left
            self.img[:self.height//2 + overlap, self.width//2 - overlap:, :], # Top right
            self.img[self.height//2 - overlap:, self.width//2 - overlap:, :]  # Bottom right
        ]
        ## Splitting annotation
        annotations = self.read_annotations()
        bottom_right, top_right, bottom_left, top_left= [], [], [], []
        for annotation in annotations.values:
            bbox_left, bbox_top, bbox_width, bbox_height, score, obj_category, truncation, occlusion = annotation
            center = (bbox_left + bbox_width//2, bbox_top + bbox_height//2)
            if center[0] > (self.width//2 - overlap) and center[1] > (self.height//2 - overlap):
                bottom_right.append([max(bbox_left - (self.width//2 - overlap), 0), max(bbox_top - (self.height//2 - overlap), 0)] + list(annotation[2:]))
            if center[0] > (self.width//2 - overlap) and center[1] < (self.height//2 + overlap):
                top_right.append([max(bbox_left - (self.width//2 - overlap), 0)] + list(annotation[1:]))
            if center[0] < (self.width//2 + overlap) and center[1] > (self.height//2 - overlap):
                bottom_left.append([bbox_left, max(bbox_top - (self.height//2 - overlap), 0)] + list(annotation[2:]))
            if center[0] < (self.width//2 + overlap) and center[1] < (self.height//2 + overlap):
                top_left.append(list(annotation))
        
        if 'images' in save_output.keys():
            for i, piece in enumerate(pieces):
                cv2.imwrite(save_output['images'] + '/' + self.id + '_' + str(i+1) + '.jpg', piece)
        if 'annotations' in save_output.keys():
            for i, sub_ann in enumerate([top_left, bottom_left, top_right, bottom_right]):
                pd.DataFrame(sub_ann).to_csv(save_output['annotations'] + '/' + self.id + '_{}.txt'.format(i+1), header=False, index=False)
        else:
            return {
                'images': pieces, 
                'annotations': [top_left, bottom_left, top_right, bottom_right]
            }
        
    def add_noise(self, coords):
        left, top, width, height = coords
        img_ = np.copy(self.img)
        img_[top:top+height, left:left+width, :] = np.random.normal(0, 1, size = (height, width, 3))
        self.img = img_
    
    def parse_annotations(self, save_output=None):
        annotations = self.read_annotations()
        if len(annotations) > 0:
            parsed = []
            for left, top, width, height, _, _, _, _ in annotations.values:
                x_center = (left + width/2) / self.width
                y_center = (top + height/2) / self.height
                width = width/self.width
                height = height/self.height
                parsed.append('0 {} {} {} {}\n'.format(x_center, y_center, width, height))
            if save_output:
                with open(os.path.join(save_output, self.id + '.txt'), 'w+') as out:
                    out.writelines(parsed)
            else:
                return parsed
        else:
            if save_output:
                open(os.path.join(save_output, self.id + '.txt'), 'w+').close()
            else:
                return []
                    