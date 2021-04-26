from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


class Image:
    def __init__(self, img_path: str) -> None:
        if not os.path.exists(img_path):
            raise FileNotFoundError('{} doest not exist'.format(img_path))  
        self.img_path_ = img_path
        self.title = os.path.basename(self.img_path_)
        self.mat = plt.imread(self.img_path_)
        self.height = self.mat.shape[0]
        self.width = self.mat.shape[1]
        self.objects = None
        self.clusters = None
    
    def plot(self, ax: object = None) -> None:
        if ax:
            ax.imshow(self.mat)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.tick_params(bottom = False, top = False, left = False, right = False, 
                            labelbottom = False, labeltop = False, labelleft = False, labelright = False)
        else:
            plt.figure(figsize=(12, 12))
            plt.imshow(self.mat)
            for spine in plt.gca().spines.values():
                spine.set_visible(False)
            plt.tick_params(bottom = False, top = False, left = False, right = False, 
                            labelbottom = False, labeltop = False, labelleft = False, labelright = False)
    
    def read_annotations(self, ann_path: str = None, subclasses: list = [1, 2], subset: list = None) -> pd.DataFrame:
        if ann_path:
            pass
        else:
            ann_path = self.img_path_.replace('images', 'annotations').replace('.jpg', '.txt')
        if os.path.exists(ann_path):
            objects = pd.read_csv(ann_path, header = None, usecols = [0, 1, 2, 3, 5], names=['left', 'top', 'width', 'height', 'object_category'])
            objects = objects[objects['object_category'].isin(subclasses)]
            objects['x'] = objects['left'] + objects['width']//2
            objects['y'] = objects['top'] + objects['height']
            objects = objects.drop(columns=['left', 'top'])
            objects = objects[['x', 'y', 'width', 'height']]
            objects = objects.reset_index(drop=True)
            if subset:
                self.objects = objects[subset]
            self.objects = objects
        else:
            raise FileNotFoundError('Annotation File Not Found!')
    
    def height_correction(self, correction_degree: int = 2) -> None:
        poly = PolynomialFeatures(degree = correction_degree)
        X = poly.fit_transform(self.objects.iloc[:, 1:2])
        y = self.objects['height'].values
        linreg = LinearRegression().fit(X, y)
        self.objects['height'] = linreg.predict(poly.transform(self.objects.iloc[:, 1:2]))
    
    def px2mm(self, dpi: int = 96) -> None:
        convert = lambda x: (x*25.4)/dpi
        self.objects = self.objects.apply(convert)
        
    def project_coordinates(self, h: int = 1750, dpi: int = 96, focal_length: int = 150):
        convert = lambda x: (x*25.4)/dpi
        # Scaling values
        self.objects['y'] = convert(self.height) - self.objects['y']
        self.objects['x'] = - convert(self.width)/2 + self.objects['x']
        # Projection
        self.objects['y'] = h*focal_length/self.objects['height']
        self.objects['x'] = self.objects['x']*self.objects['y']/focal_length