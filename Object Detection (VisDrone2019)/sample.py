from dataset.processing import read_image
import dataset.navigation as nav
import numpy as np
import os

samples = np.random.choice(os.listdir(os.path.join(nav.VALID_PATH, 'images')), size=5, replace=False)
samples = [os.path.join(nav.VALID_PATH, 'images', sample) for sample in samples]
samples = [read_image(sample) for sample in samples]


# for sample in samples: sample.display()

for sample in samples:
    sample.pad()
    locs = ['upper left', 'lower left', 'upper right', 'lower right']
    if sample.height == 736:
        locs = ['upper left', 'upper right']
    if sample.width == 736:
        locs = ['upper left', 'lower left']
    for loc in locs:
        crop_ = sample.crop(loc)
        crop_.save(output_path = 'trash', ann_sep = ',')