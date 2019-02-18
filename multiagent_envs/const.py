import numpy as np

X = np.array([1, 0])
Y = np.array([0, 1])
Z = np.array([0, 0, 1])
O = np.array([0, 0])

UP, LEFT, STAY, DOWN, RIGHT = -1, -.5, 0, .5, 1

DIRECTIONS = {UP: Y, LEFT: -X, STAY: O, DOWN: -Y, RIGHT: X}
