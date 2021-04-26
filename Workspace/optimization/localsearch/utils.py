import numpy as np

dist = lambda a, b: np.sqrt(np.sum((a - b)**2))

# def cost_func(route: np.ndarray) -> float:
#     return (
#         dist(np.array([0, 0]), route[0, :]) +
#         np.sum([dist(route[i, :], route[i + 1, :]) for i in range(len(route) - 1)]) + 
#         dist(route[-1, :], np.array([0, 0]))
#     )

def swap(route: np.ndarray, i: int, j: int) -> np.ndarray:
    route_ = np.zeros(route.shape) - 1
    route_[:i, :] = route[:i, :]
    route_[i:j+1, :] = np.flip(route[i:j+1, :], axis=0)
    route_[j+1:, :] = route[j+1:, :]
    return route_

def cost_func(route: np.ndarray, alpha: float) -> float:
    dist_ = (
        dist(np.array([0, 0]), route[0, :-1]) +
        np.sum([dist(route[i, :-1], route[i + 1, :-1]) for i in range(len(route) - 1)]) + 
        dist(route[-1, :-1], np.array([0, 0]))
    )
    risk_ = np.sum(np.abs(np.array(sorted(route[:, -1], reverse=True)) - route[:, -1]))
    return alpha*risk_ + (1 - alpha)*dist_