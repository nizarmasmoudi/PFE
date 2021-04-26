from optimization.localsearch.utils import cost_func
from optimization.localsearch.algorithm import OPT2
from optimization.evolutionary.algorithm import GA
from clustering.operations import Clustering
from clustering.image import Image
from itertools import permutations
import dataset.navigation as nav
import matplotlib.pyplot as plt
import numpy as np
import random
import scipy
import os

# Paremeters
N_CLUSTERS = 5
ALPHA = .5

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

# Scoring clusters
clusters_ = cls.image.objects[['x', 'y']].copy()
clusters_['label'] = cls.labels_
clusters_ = clusters_.loc[clusters_['label'] > -1]
clusters_ = clusters_.reset_index(drop = True)
clusters = clusters_.groupby('label').mean()
for c in clusters.index:
    dists = scipy.spatial.distance.cdist(clusters_.loc[clusters_['label'] == c].values, clusters_.loc[clusters_['label'] == c].values)
    clusters.loc[c, 'score'] = (len(set(dists[(dists < 2000) & (dists > 0)])) / len(set(dists[dists > 0]))) + len(clusters_.loc[clusters_['label'] == c])
clusters = clusters.values

# Optimization (Brute force)
combinations = list(permutations(range(len(clusters))))
combinations = [clusters[combination, :] for combination in combinations]
costs = [cost_func(combination, ALPHA) for combination in combinations]
bf_route = combinations[np.argmin(costs)]

# Optimization (2-Opt)
alg = OPT2(alpha = ALPHA)
alg.fit(clusters)
opt_route = alg.route_

# Optimization (Genetic algorithm)
# alg = GA(population_size = 10, mutation_rate = .05, crossover_rate = 1)
# alg.fit(clusters, n_generations = 20, verbose = False)
# ga_route = alg.population[np.argmin([cost_func(ind.xy, alpha = .8) for ind in alg.population])]

# Visualisation
fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)

ax1.scatter(clusters[:, 0], clusters[:, 1], color='blue', s=200, alpha=.6)
ax1.scatter([0], [0], color='blue', s=200, alpha=.6)
ax1.plot(bf_route[:, 0], bf_route[:, 1], color='grey', lw=1, linestyle='--', alpha=.6)
ax1.plot([0, bf_route[0, 0]], [0, bf_route[0, 1]], color='grey', lw=1, linestyle='--', alpha=.6)
# ax1.plot([0, bf_route[-1, 0]], [0, bf_route[-1, 1]], color='grey', lw=1, linestyle='--', alpha=.6)
ax1.set_title(f'Brute Force Route\nDistance = {cost_func(bf_route, 0)/1000:.1f}\nOrder Error = {cost_func(bf_route, 1)}', fontsize='small')

ax2.scatter(clusters[:, 0], clusters[:, 1], color='blue', s=200, alpha=.6)
ax2.scatter([0], [0], color='blue', s=200, alpha=.6)
ax2.plot(opt_route[:, 0], opt_route[:, 1], color='grey', lw=1, linestyle='--', alpha=.6)
ax2.plot([0, opt_route[0, 0]], [0, opt_route[0, 1]], color='grey', lw=1, linestyle='--', alpha=.6)
# ax2.plot([0, opt_route[-1, 0]], [0, opt_route[-1, 1]], color='grey', lw=1, linestyle='--', alpha=.6)
ax2.set_title(f'2-Opt Route\nDistance = {cost_func(opt_route, 0)/1000:.1f}\nOrder Error = {cost_func(opt_route, 1)}', fontsize='small')

for ax in [ax1, ax2]:
    ax.tick_params(bottom = False, top = False, left = False, right = False, labelbottom = False, labeltop = False, labelleft = False, labelright = False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for x, y, r in clusters[:, :]:
        ax.text(x, y, s=str(int(r)), va='center', ha='center', color='white', alpha=.8, fontsize='medium')
        ax.text(0, 0, s='UAV', va='center', ha='center', color='white', alpha=.8, fontsize='xx-small')

plt.show()