import numpy as np
from .utils import swap, cost_func
from typing import Callable

class OPT2:
    def __init__(self, cost_func: Callable[[np.ndarray], float] = cost_func, **kwargs) -> None:
        self.route_ = None
        self.cost_ = -1
        self.cost_func = cost_func
        self.alpha = kwargs['alpha'] if 'alpha' in kwargs.keys() else None
    def fit(self, data: np.ndarray) -> None:
        self.route_ = data.copy()
        self.cost_ = cost_func(self.route_, self.alpha)# if self.alpha else cost_func(self.route_)
        while True:
            improved = False
            for i in range(1, len(self.route_) - 1):
                for j in range(i + 1, len(self.route_)):
                    route_ = swap(self.route_, i, j)
                    cost = cost_func(route_, self.alpha)# if self.alpha else cost_func(route_)
                    if cost < self.cost_:
                        self.route_ = route_.copy()
                        self.cost_ = cost
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
            