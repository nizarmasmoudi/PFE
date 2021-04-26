from clustering.operations import Clustering
from clustering.image import Image
import dataset.navigation as nav
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os

# Picking a random image
# paths = os.listdir(os.path.join(nav.VALID_PATH, 'images'))
# paths = [os.path.join(nav.VALID_PATH, 'images', path) for path in paths]
# path = random.choices(paths, k = 1)[0]
path = r'C:\Users\nizar\OneDrive\Documents\VisDrone2019\validation\images\0000069_00001_d_0000001.jpg'

# Making an image object
image = Image(path)
image.read_annotations()

# Clustering
cls = Clustering(
    image = image, 
    focal_length = 150, 
    correction_degree = 2, 
    dpi = 96, 
    distance_threshold = 2000
)
cls.fit()

# Visudalisation
colors = ['red', 'blue', 'green', 'orange', 'yellow', 'magenta', 'violet', 'cyan', 'white', 'pink', 'turquoise', 'olive', 'lime', 'sienna', 'black']
ax = plt.subplot()
image.plot(ax = ax)
plot = sns.scatterplot(data = image.objects, x = 'x', y = 'y', hue = cls.labels_, sizes = 100, palette = colors[-1:] + colors[:len(set(cls.labels_)) - 1], alpha=.6)
for t, l in zip(plot.get_legend().texts, ['Safe'] + [f'Cluster {i + 1}' for i in range(len(set(cls.labels_)))]): t.set_text(l)
plt.show()