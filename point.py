import matplotlib.pyplot as plt
import numpy as np


class Point:
    """Index, cartesian coordinates in 2D and intrinsic direction and speedx&y

    >>> c = Point(1, 2, 5, 10, 1)
    >>> c.x
    1
    >>> c.y
    2
    >>> c.index
    5
    >>> c.speedx
    10
    >>> c.speedy
    1
    """

    def __init__(self, index, x0, y0, speedx, speedy):
        self._index = index
        self._x0 = x0
        self._y0 = y0
        self._speedx = speedx
        self._speedy = speedy

    @property
    def x0(self):
        return self._x0

    @property
    def y0(self):
        return self._y0

    @property
    def index(self):
        return self._index

    @property
    def speedx(self):
        return self._speedx

    @property
    def speedy(self):
        return self._speedy