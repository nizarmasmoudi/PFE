# %% Imports
import numpy as np
from itertools import product
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import networkx as nx
import os
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.cluster import DBSCAN
from time import time

from utils import env
from dataset.image import Image

from jupyterthemes import jtplot
jtplot.style('onedork')
jt_colors = ['#ea8080', '#61afef', '#94c273', '#fea47f', '#DDD7A3', '#e07a7a', '#cc89e0', '#51b6c3', '#6a737d']# red blue green orange yellow magenta violet cyan grey
colors = ['red', 'blue', 'green', 'orange', 'yellow', 'magenta', 'violet', 'cyan', 'white', 'pink', 'turquoise', 'olive', 'lime', 'sienna', 'black']
# %% Global parameters
start = time()
IMG_PATH = os.path.join(env.VALID_PATH, 'images', '0000069_00001_d_0000001.jpg')
H = 1750
FOCAL_LENGTH = 150
CORRECTION_DEGREE = 2
DPI = 96
DISTANCE_THRESHOLD = 2000
print('--- Execution time --- {:.4f} seconds'.format(time() - start))
# %% Reading image and annotations
start = time()
image = Image(IMG_PATH)
objects = image.read_annotations(subset=['left', 'top', 'width', 'height'])
objects['x'] = objects['left'] + objects['width']//2
objects['y'] = objects['top'] + objects['height']
objects = objects.drop(columns=['left', 'top'])
objects = objects[['x', 'y', 'width', 'height']]
print('--- Execution time --- {:.4f} seconds'.format(time() - start))
# %% Reading image and annotations (visualisation)
fig, ax = plt.subplots(figsize=(15, 10))
image.show(ax = ax)
for x, y, width, height in objects.values:
    left, top = x - width//2, y - height
    rect = Rectangle((left, top), width, height, linewidth = 1, edgecolor='navy', facecolor='none')
    ax.add_patch(rect)
ax.grid(False)
# %% Height correction
start = time()
objects = objects.sort_values(by = 'y')
poly = PolynomialFeatures(degree = CORRECTION_DEGREE)
X = poly.fit_transform(objects.iloc[:, 1:2])
reg = LinearRegression().fit(X, objects['height'].values)
objects['height_c'] = reg.predict(poly.transform(objects.iloc[:, 1:2]))
print('--- Execution time --- {:.4f} seconds'.format(time() - start))
# %% Height correction (visualisation)
fig, ax = plt.subplots(figsize=(15, 10))
image.show(ax = ax)
for x, y, width, height, height_c in objects.values:
    left, top_c, top = x - width//2, y - height_c, y - height
    rect1 = Rectangle((left, top), width, height, linewidth=1, edgecolor='red', alpha=1, facecolor='none')
    rect2 = Rectangle((left, top_c), width, height_c, linewidth=1, edgecolor='green', alpha=1, facecolor='none')
    ax.add_patch(rect1)
    ax.add_patch(rect2)
ax.grid(False) 
# %% Pixel to metric system conversion
start = time()
convert = lambda x: (x*25.4)/DPI
objects_ = objects.copy()
objects_ = objects_.apply(convert)
print('--- Execution time --- {:.4f} seconds'.format(time() - start))
# %% Re-positioning objects
start = time()
objects_['y'] = convert(image.height) - objects_['y']
objects_['x'] = - convert(image.width)/2 + objects_['x']
objects_['y_c'] = H*FOCAL_LENGTH/objects_['height_c']
objects_['x_c'] = objects_['x']*objects_['y_c']/FOCAL_LENGTH
print('--- Execution time --- {:.4f} seconds'.format(time() - start))
# %% Re-positioning objects (visualisation)
fig, ax = plt.subplots(figsize=(12, 8))
fig.suptitle('Re-positioned objects ($f = FOCAL_LENGTH$ mm)'.replace('FOCAL_LENGTH', str(FOCAL_LENGTH)))
ax.scatter(x = objects_.apply(lambda v: v/1000)['x_c'], y = objects_.apply(lambda v: v/1000)['y_c'], s=50, color=jt_colors[1])
for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(
    bottom = False, top = False, left = False, right = False, 
    labelbottom = True, labeltop = False, labelleft = True, labelright = False
)
ax.set_ylabel('Longitudinal distance from camera (m)')
ax.set_xlabel('Latitudinal distance from camera (m)')
ax.grid(True)
# %% Clustering
start = time()
model = DBSCAN(eps = DISTANCE_THRESHOLD, min_samples = 3).fit(objects_.loc[:, ['x_c', 'y_c']])
print('--- Execution time --- {:.4f} seconds'.format(time() - start))
# %% Clustering (visualisation)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 20))
image.show(ax = ax1)
ax1.scatter(x = objects['x'], y = objects['y'], color = [colors[i] for i in model.labels_], alpha=.6, s=70)
ax1.grid(False)
ax1.legend(handles=[
        Line2D([0], [0], marker='o', color=colors[k], lw=0, label='Safe' if k == -1 else 'Cluster {}'.format(k + 1), markersize=8, alpha=.6) for k in set(model.labels_)
    ], loc='lower right', fontsize='x-small', framealpha=.3, ncol=3)
ax1.set_title('Clusters displayed on source image', fontdict = {'fontsize': 20})
ax2.scatter(x = objects_['x_c'].apply(lambda x: x/1000), y = objects_['y_c'].apply(lambda y: y/1000), s=70, color = [colors[i] for i in model.labels_], alpha=.8)
for spine in ax2.spines.values():
    spine.set_visible(False)
ax2.tick_params(
    bottom = False, top = False, left = False, right = False, 
    labelbottom = True, labeltop = False, labelleft = True, labelright = False
)
ax2.set_ylabel('Longitudinal distance from camera (m)')
ax2.set_xlabel('Latitudinal distance from camera (m)')
ax2.set_title('Clusters displayed on re-positioned objects', fontdict = {'fontsize': 20})
ax2.legend(handles=[
        Line2D([0], [0], marker='o', color=colors[k], lw=0, label='Safe' if k == -1 else 'Cluster {}'.format(k + 1), markersize=8, alpha=.6) for k in set(model.labels_)
    ], loc='upper left', fontsize='small', framealpha=.3, ncol=3)
plt.show()
# %% Scoring clusters
start = time()
dist = lambda A, B: np.sqrt(np.sum((A - B)**2))
objects_['cluster'] = model.labels_
clusters = pd.DataFrame(index=objects_.loc[objects_['cluster'] > -1, 'cluster'].unique())
for c in clusters.index:
    cluster_objects = objects_.loc[objects_['cluster'] == c, ['x_c', 'y_c']]
    dist_inf_2 = 0
    for o in cluster_objects.index:
        dists = np.array([cluster_objects.loc[o:].apply(lambda O: dist(O.values, cluster_objects.loc[o]), axis=1)])
        dist_inf_2 += len(dists[(dists < 2000) & (dists > 0)])
    clusters.loc[c, 'outage_probability'] = dist_inf_2/np.sum(range(len(cluster_objects)))
    clusters.loc[c, 'risk'] = clusters.loc[c, 'outage_probability'] + len(cluster_objects)
    clusters.loc[c, 'x_centroid'] = cluster_objects.mean(axis=0).loc['x_c']
    clusters.loc[c, 'y_centroid'] = cluster_objects.mean(axis=0).loc['y_c']
clusters.loc[:, 'centroid'] = clusters.loc[:, ['x_centroid', 'y_centroid']].apply(lambda row: row.values, axis=1)
clusters = clusters.drop(columns=['x_centroid', 'y_centroid'])
print('--- Execution time --- {:.4f} seconds'.format(time() - start))
# %% Scoring clusters (visualisation)
fig, ax = plt.subplots(figsize=(12, 10))
tmp = objects_.loc[objects_['cluster'] > -1, ['x_c', 'y_c', 'cluster']].copy()
tmp.loc[:, ['x_c', 'y_c']] = tmp.loc[:, ['x_c', 'y_c']].apply(lambda v: v/1000)
clusters_ = clusters.sort_values(by='risk', ascending=False)
ax.scatter(x = tmp['x_c'], y = tmp['y_c'], s=100, color = [colors[i] for i in tmp['cluster']])
ax.set_xticks(range(tmp['x_c'].astype(int).min() - 5, tmp['x_c'].astype(int).max() + 5, 5))
ax.set_yticks(range(0, tmp['y_c'].astype(int).max() + 5, 5))
for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(
    bottom = False, top = False, left = False, right = False, 
    labelbottom = True, labeltop = False, labelleft = True, labelright = False
)
ax.legend(handles=[Line2D(
    [0], [0],
    marker='o', color=colors[c], lw=0, markersize=10, alpha=.6,
    label='Cluster {}'.format(c + 1) + ': Outage probability = {outage_probability}, Risk = {risk})'.format_map(clusters.loc[c][0:]) 
) for c in clusters.sort_values(by='risk').index], loc='upper left', fontsize='medium', framealpha=.3, ncol=1)
plt.show()
# %% UAV trajectory (visualisation)
fig, ax = plt.subplots(figsize=(12, 10))

options = {'node_size': 700, 
           'alpha': 0.6,
           'node_color':colors[:len(clusters)] + ['grey'],}

G = nx.DiGraph()
pos = {i:clusters.loc[i, 'centroid'] for i in clusters.index}
pos[-1] = [0, 0]

for i in clusters.sort_index().index:
    G.add_node(i, risk=clusters.loc[i, 'risk'])
G.add_node(-1, risk=0)

for i in clusters.index:
    G.add_weighted_edges_from([(i, j, dist(clusters.loc[i, 'centroid']/1000, clusters.loc[j, 'centroid']/1000)) for j in clusters.index if i!=j])
    G.add_edge(-1, i, weight=dist(clusters.loc[i, 'centroid']/1000, [0, 0]))

nx.draw_networkx(G, pos, with_labels=False, width=2, **options)

edge_labels = {i:round(w) for i, w in nx.get_edge_attributes(G,'weight').items()}
nx.draw_networkx_edge_labels(G, pos, edge_labels = edge_labels, bbox=dict(alpha=0), font_size=20, verticalalignment='bottom')
node_labels = {i:round(w, 1) for i, w in nx.get_node_attributes(G,'risk').items()}
nx.draw_networkx_labels(G, pos, labels = node_labels, font_size=12, font_color='white', alpha=.6, font_weight='bold')

handles = [Line2D([0], [0], marker='o', color=colors[c], lw=0, markersize=10, alpha=.6, label='Cluster {}'.format(c + 1)) for c in clusters.sort_index().sort_values(by='risk').index]
handles = handles + [Line2D([0], [0], marker='o', color='grey', lw=0, markersize=10, alpha=.6, label='UAV')]
ax.legend(handles=handles, loc='lower left', fontsize='medium', framealpha=.3, ncol=1)
ax.set_title('UAV trajectory', fontsize=20)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(
    bottom = False, top = False, left = False, right = False, 
    labelbottom = False, labeltop = False, labelleft = False, labelright = False
)
ax.grid(False)
plt.show()
# %% UAV trajectory (low to high risk)
print('Path is (higher to lower risk)', ' --> '.join(clusters.reset_index().sort_values('risk', ascending=False)['index'].astype(str)))
print('Path is (higher to lower outage probability)', ' --> '.join(clusters.reset_index().sort_values('outage_probability', ascending=False)['index'].astype(str)))
# %% TSP resolution
def get_trajectory(source=-1):
    root = [source]
    while len(root[1:]) < len(clusters):
        neighbors = [n for n in G.neighbors(source) if n not in root]
        _, next = min([(G[source][n]['weight']/G._node[n]['risk'], n) for n in neighbors])
        source = next
        root.append(source)
    return root
print('trajectory is', get_trajectory())
# %%
