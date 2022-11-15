import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from typing import Tuple


class World(object):
    """"""

    MAP_VALUES = {
        '~': 800,
        '*': 200,
        '+': 150,
        'X': 120,
        '_': 100,
        'H': 70,
        'T': 50,
    }

    def __init__(self, height: int, width: int, **kwargs) -> None:
        # Validate inputs
        if height <= 0 or not isinstance(height, int):
            raise ValueError('Value for map height must be a positive integer.')

        if width <= 0 or not isinstance(width, int):
            raise ValueError('Value for map width must be a positive integer.')

        self.h = height
        self.w = width
        self.map = None

    def add_world_terrain(self, string_terrain: str) -> None:
        map_values = [self.MAP_VALUES.get(char, 9999) for char in string_terrain.replace('\n', '')]

        self.map = np.array(map_values).reshape((self.h, self.w))

    def dijkstra(self, start: Tuple[int, int], finish: Tuple[int, int]) -> float:
        ...

    def view_map(self):
        if self.map is None:
            raise AttributeError('World not initialized yet!')

        view_map = self.map
        # view_map[view_map == 9999] = 0
        sns.heatmap(view_map)
        plt.show()

    def __str__(self) -> str:
        if self.map is None:
            return 'World not initialized yet!'

        string = ''
        for row in self.map:
            string += '|'
            for item in row:
                string += f'{item:^6}|'
            string += '\n'
        return string
