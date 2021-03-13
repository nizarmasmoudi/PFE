import matplotlib.pyplot as plt
import os
import pandas as pd

class Image:
    def __init__(self, img_path):
        if not os.path.exists(img_path):
            raise FileNotFoundError('{} doest not exist'.format(img_path))  
        self.img_path = img_path
        self.id = self.img_path.split('\\')[-1].split('.')[0]
        self.mat = plt.imread(self.img_path)
        self.height = self.mat.shape[0]
        self.width = self.mat.shape[1]
        self.ann_path = self.img_path.replace('images', 'annotations').replace('.jpg', '.txt')
    
    def show(self, ax=None):
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
    
    def read_annotations(self, subclasses=[1, 2], subset=None):
        if os.path.exists(self.ann_path):
            df = pd.read_csv(self.ann_path, header = None, names=['left', 'top', 'width', 'height', 'score', 'object_category', 'truncation', 'occlusion', '_'])
            df.dropna(axis = 'columns', inplace=True)
            df = df[df['object_category'].isin(subclasses)]
            if subset:
                return df[subset]
            return df
        else:
            raise FileNotFoundError('Annotation File Not Found!')
        
    def describe(self):
        desc = pd.DataFrame()
        desc['Id'] = [self.id]
        desc['Height'] = [self.height]
        desc['Width'] = [self.width]
        print(desc.set_index('Id'))