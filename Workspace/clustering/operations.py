from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.cluster import DBSCAN
from clustering.image import Image
from copy import deepcopy
import numpy as np

class Clustering:
    def __init__(self, image: Image, focal_length: int = 150, correction_degree: int = 2, dpi: int = 96, distance_threshold: int = 2000) -> None:
        self.image = deepcopy(image)
        self.focal_length = focal_length
        self.correction_degree = correction_degree
        self.dpi = dpi
        self.distance_threshold = distance_threshold
        self.labels_ = None
    def fit(self):
        self.image.height_correction(correction_degree = self.correction_degree)
        self.image.px2mm(dpi = self.dpi)
        self.image.project_coordinates(dpi = self.dpi, focal_length = self.focal_length, h = 1750)
        model = DBSCAN(eps = self.distance_threshold, min_samples = 3).fit(self.image.objects.loc[:, ['x', 'y']])
        self.labels_ = model.labels_