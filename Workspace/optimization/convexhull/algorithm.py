import numpy as np

class JarvisMarch:
    def __init__(self) -> None:
        pass
    def fit(self, points) -> np.ndarray:
        def find_next(anchor, points):
            points = np.array([point for point in points if not np.array_equal(point, anchor)])
            cursor = points[np.random.randint(0, len(points)), :]
            for point in points:
                z = np.cross(cursor - anchor, point - anchor)
                if z > 0:
                    cursor = point
            return cursor

        points = points[np.lexsort((points[:, 1], points[:, 0]))]
        left_most = points[0, :]
        next_point = find_next(left_most, points)
        hull = [left_most, next_point]
        while True:
            next_point = find_next(next_point, points)
            hull.append(next_point)
            if np.array_equal(next_point, left_most):
                break
        return np.array(hull)