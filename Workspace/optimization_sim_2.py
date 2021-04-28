import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA
from optimization.convexhull.algorithm import JarvisMarch
from time import time
plt.style.use('seaborn')

points = np.random.randint(0, 10, size = (5, 2))

points = np.array([
    [1, 1],
    [2, 1.3],
    [3, 1.2],
    [4, 1.1],
])

t0 = time()
hull = JarvisMarch().fit(points)
print(f'--- Execution time --- {time() - t0:.4f} seconds')
print(hull)

k = 2
center = np.mean(hull[:-1, :], axis = 0)
traj = ((hull - center)/LA.norm((hull - center), axis = 1, keepdims=True))*(LA.norm((hull - center), axis = 1, keepdims=True) + k) + center

plt.figure()
plt.scatter(x = points[:, 0], y = points[:, 1])
# plt.scatter(x = center[0], y = center[1])
plt.plot(hull[:, 0], hull[:, 1], color='grey', lw=1, linestyle='--', label='Convex Hull')
plt.plot(traj[:, 0], traj[:, 1], color='purple', lw=1, linestyle='--', label='UAV trajectory')
plt.show()