from __future__ import annotations

from collections import namedtuple
from functools import lru_cache
from itertools import chain

import numpy as np
from numpy import double
from numpy.typing import ArrayLike, NDArray

from .utility import rand

# 座標
Coord = namedtuple("Coord", ["x", "y"], defaults=(double(0), double(0)))


class Location(object):
    """
    一個城市的座標位置
    """

    def __init__(self, x, y=None) -> None:
        if y is not None:
            _x, _y = x, y
        elif isinstance(x, np.complexfloating):
            _x, _y = x.real, x.imag
        elif isinstance(x, Location) or isinstance(x, Coord):
            _x, _y = x.x, x.y
        elif isinstance(x, list) or isinstance(x, ArrayLike):
            _x, _y = x[:2]
        else:
            raise ValueError(f"Unknown type: {type(x)} {x}")

        if not isinstance(_x, double):
            _x = double(_x)
        if not isinstance(_y, double):
            _y = double(_y)

        self._x, self._y = _x, _y

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return f"<Location at ({self._x:.4f}, {self._y:.4f})>"

    def __json__(self):
        return [self._x, self._y]
        # return {
        #     'x': self._x,
        #     'y': self._y
        # }

    def __eq__(self, other: Location) -> bool:
        return self.x == other.x and self.y == other.y

    def __sub__(self, other: Location):
        """
        與另一個地點之間的距離
        """
        dx, dy = self.x - other.x, self.y - other.y
        return np.sqrt(dx ** 2 + dy ** 2)

    @property
    def x(self) -> double:
        return self._x

    @property
    def y(self) -> double:
        return self._y

    @property
    def coord(self) -> tuple[double, double]:
        return (self._x, self._y)


class Route(object):
    def __init__(self, path: ArrayLike[Location] | Route) -> None:
        """
        一條路徑

        Args:
            path (list[Location]): 路徑上的城市
        """
        if isinstance(path, Route):
            self._path_setter(path.path)
        elif isinstance(path, ArrayLike):
            self._path_setter(path)
        else:
            raise ValueError(f"Unknown type: {type(path)}")

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return f'<Route distance={self._distance}>'

    def __json__(self):
        return {
            'path': self._path,
            'distance': self._distance
        }

    def __getitem__(self, key: int) -> Location:
        return self._path[key]

    def __len__(self) -> int:
        return len(self._path)

    def __iter__(self):
        return iter(self._path)

    def __lt__(self, other: Route) -> bool:
        return self._distance < other._distance

    @property
    def path(self) -> NDArray:
        return self._path

    @path.setter
    def path(self, path: ArrayLike[Location]) -> None:
        self._path_setter(path)

    def _path_setter(self, path: ArrayLike[Location]) -> None:
        self._path = np.array(path)
        # 計算兩點之間的距離，最後還要回到起點
        diff = np.abs(np.diff(self._path, append=self._path[0]))
        self._distance = np.sum(diff)

    @property
    def distance(self) -> double:  # 路徑總長
        return self._distance

    @classmethod
    def random(cls, locations: list[Location]) -> Route:
        """
        隨機生成一條路徑

        Args:
            locations (list[Location]): 所有城市的座標

        Returns:
            Route: 一條隨機路徑
        """
        start, path = np.split(locations, [1])
        rand.shuffle(path)
        path = np.concatenate((start, path))
        return cls(path)

    @classmethod
    def crossover(cls, f1: Route, f2: Route, k: int, mutation_rate: float) -> Route:
        """
        交配

        Args:
            f1 (Route): 父代 1
            f2 (Route): 父代 2
            k (int): 基因要切成幾段
            mutation_rate (float): 突變率

        Returns:
            Route: 子代
        """
        g1 = np.array_split(f1.path, k)[::2]  # 將 F1 切成 k 段，取偶數段
        _tmp = np.concatenate(g1)

        g2 = np.setdiff1d(f2.path, _tmp, assume_unique=True)  # F2 取不在 F1 的城市
        g2 = np.array_split(g2, k // 2)  # 將 F2 切成 k / 2 段

        ch = list(chain.from_iterable(zip(g1, g2)))

        if len(g1) > len(g2):  # g1 比 g2 長 (k 為奇數)
            ch.append(g1[-1])

        ch = cls(np.concatenate(ch))

        # if len(ch) != len(f1):
        #     print(f'G1: {[len(g) for g in g1]}\nG2: {[len(g)
        #           for g in g2]}\nF1: {len(f1)}, F2: {len(f2)}, CH: {len(ch)}')

        ch.mutate(mutation_rate)

        return ch

    def mutate(self, mutation_rate: float):
        """
        突變

        Args:
            mutation_rate (float): 突變率

        Returns:
            Route: 突變後的路徑
        """
        n = len(self._path)
        for i, r in np.ndenumerate(rand.random(n)):
            i = i[0]
            if i != 0 and r <= mutation_rate:  # 起點不能改變
                j = rand.integers(1, n)
                self._path[[i, j]] = self._path[[j, i]]
