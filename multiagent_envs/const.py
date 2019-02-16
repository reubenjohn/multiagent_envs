import numpy as np

X = np.array([1, 0])
Y = np.array([0, 1])
Z = np.array([0, 0, 1])

UP, LEFT, DOWN, RIGHT = 0, 1, 2, 3

DIRECTIONS = {UP: Y, LEFT: -X, DOWN: -Y, RIGHT: X}