from .metrics import cost_func
import numpy as np

class Individual:
    def __init__(self, xyc: np.ndarray) -> None:
        self.xyc = xyc
        self.length = len(xyc)
    def __repr__(self) -> str:
        return '(0, 0) --> ' + ' --> '.join([f'{(c[0], c[1], c[2])}' for c in self.xyc]) + ' --> (0, 0)'
    def fitness(self, alpha: float) -> float:
        return 1/cost_func(self.xyc, alpha)
    def mutate(self, p: float) -> None:
        if np.random.random() < p:
            i, j = np.random.randint(0, self.length), np.random.randint(0, self.length)
            self.xyc[[i, j], :] = self.xyc[[j, i], :]
            # i = np.random.randint(0, self.length - 1)
            # self.xyc[[i, i + 1], :] = self.xyc[[i + 1, i], :]
    def __eq__(self, o: 'Individual') -> bool:
        return np.array_equal(self.xyc, o)
    @staticmethod
    def crossover(parents: list) -> list:
        i = np.random.randint(0, parents[0].length - 1)
        j = np.random.randint(i + 1, parents[0].length)
        children_xyc = [np.zeros(parents[0].xyc.shape, dtype=np.int) - 1, np.zeros(parents[0].xyc.shape, dtype=np.int) - 1]
        for k in range(2):
            children_xyc[k][i:j, :] = parents[k].xyc[i:j, :]
            mask = np.invert(np.sum([(parents[1 - k].xyc == ind).all(axis=1) for ind in children_xyc[k]], axis=0, dtype=bool))
            children_xyc[k][:i, :] = parents[1 - k].xyc[mask, :][:i, :]
            children_xyc[k][j:, :] = parents[1 - k].xyc[mask, :][i:, :]
        return [Individual(children_xyc[0]), Individual(children_xyc[1])]