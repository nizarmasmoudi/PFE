from optimization.localsearch.utils import cost_func
from optimization.localsearch.algorithm import OPT2
from optimization.evolutionary.algorithm import GA
import matplotlib.gridspec as gridspec
from itertools import permutations
import matplotlib.pyplot as plt
import numpy as np
from time import time

# Paremeters
N_CLUSTERS = 7
ALPHA = .9

# Simulating clusters
clusters = np.random.randint(1, 100, size = [N_CLUSTERS, 2])
clusters = np.concatenate((clusters, np.random.randint(1, 10, size = [N_CLUSTERS, 1])), axis=1)

# Optimization (Brute force)
if N_CLUSTERS < 8:
    bf_t0 = time()
    combinations = list(permutations(range(N_CLUSTERS)))
    combinations = [clusters[combination, :] for combination in combinations]
    costs = [cost_func(combination, ALPHA) for combination in combinations]
    bf_route = combinations[np.argmin(costs)]
    bf_t = time() - bf_t0

# Optimization (2-Opt)
opt_t0 = time()
alg = OPT2(alpha = ALPHA)
alg.fit(clusters)
opt_route = alg.route_
opt_t = time() - opt_t0

# Optimization (Genetic algorithm)
ga_t0 = time()
alg = GA(population_size = 20, mutation_rate = .05, crossover_rate = .8)
history = alg.fit(clusters, n_generations = 50, alpha = ALPHA, verbose = False)
ga_route = alg.population[np.argmin([cost_func(ind.xyc, alpha = 0) for ind in alg.population])]
ga_t = time() - ga_t0

# Visualisation
fig = plt.figure()
gs = gridspec.GridSpec(nrows = 2, ncols = 2, figure=fig)
ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1])

if N_CLUSTERS < 8:
    ax1.scatter(clusters[:, 0], clusters[:, 1], color='red', s=150, alpha=.6, clip_on = False)
    ax1.scatter([0], [0], color='red', s=150, alpha=.6, clip_on = False)
    ax1.plot(bf_route[:, 0], bf_route[:, 1], color='grey', lw=1, linestyle='--', alpha=.6)
    ax1.plot([0, bf_route[0, 0]], [0, bf_route[0, 1]], color='grey', lw=1, linestyle='--', alpha=.6)
    ax1.plot([0, bf_route[-1, 0]], [0, bf_route[-1, 1]], color='grey', lw=1, linestyle='--', alpha=.6)
    ax1.set_title(f'Brute Force Route\nDistance = {cost_func(bf_route, 0):.1f}, Order Error = {cost_func(bf_route, 1)}\n Execution time = {bf_t:.5f} seconds', fontsize='x-small', color='red')
else:
    ax1.set_title(f'Brute Force Route\nDistance = {None}\nOrder Error = {None}', fontsize='x-small', color='red')
    ax1.text(.5, .5, f'Number of clusters is too large ({N_CLUSTERS})', va='center', ha='center', fontsize='medium', color='red')
    ax1.set_xlim(xmin=0, xmax=1)
    ax1.set_ylim(ymin=0, ymax=1)

ax2.scatter(clusters[:, 0], clusters[:, 1], color='blue', s=150, alpha=.6, clip_on = False)
ax2.scatter([0], [0], color='blue', s=150, alpha=.6, clip_on = False)
ax2.plot(opt_route[:, 0], opt_route[:, 1], color='grey', lw=1, linestyle='--', alpha=.6)
ax2.plot([0, opt_route[0, 0]], [0, opt_route[0, 1]], color='grey', lw=1, linestyle='--', alpha=.6)
ax2.plot([0, opt_route[-1, 0]], [0, opt_route[-1, 1]], color='grey', lw=1, linestyle='--', alpha=.6)
ax2.set_title(f'2-Opt Route\nDistance = {cost_func(opt_route, 0):.1f}, Order Error = {cost_func(opt_route, 1)}\n Execution time = {opt_t:.5f} seconds', fontsize='x-small', color='blue')

ax3.scatter(clusters[:, 0], clusters[:, 1], color='green', s=150, alpha=.6, clip_on = False)
ax3.scatter([0], [0], color='green', s=150, alpha=.6, clip_on = False)
ax3.plot(ga_route.xyc[:, 0], ga_route.xyc[:, 1], color='grey', lw=1, linestyle='--', alpha=.6)
ax3.plot([0, ga_route.xyc[0, 0]], [0, ga_route.xyc[0, 1]], color='grey', lw=1, linestyle='--', alpha=.6)
ax3.plot([0, ga_route.xyc[-1, 0]], [0, ga_route.xyc[-1, 1]], color='grey', lw=1, linestyle='--', alpha=.6)
ax3.set_title(f'GA Route\nDistance = {cost_func(ga_route.xyc, 0):.1f}, Order Error = {cost_func(ga_route.xyc, 1)}\n Execution time = {ga_t:.5f} seconds', fontsize='x-small', color='green')

# for ax in [ax1, ax2, ax3]:
for ax in [ax1, ax2, ax3]:
    ax.tick_params(bottom = False, top = False, left = False, right = False, labelbottom = False, labeltop = False, labelleft = False, labelright = False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for x, y, r in clusters[:, :]:
        ax.text(x, y, s=str(r), va='center', ha='center', color='white', alpha=.8, fontsize='medium')
        ax.text(0, 0, s='UAV', va='center', ha='center', color='white', alpha=.8, fontsize='xx-small')

plt.show()