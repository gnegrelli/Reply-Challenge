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
        map_values = [self.MAP_VALUES.get(char, -1) for char in string_terrain.replace('\n', '')]

        self.map = np.array(map_values).reshape((self.h, self.w))

    def dijkstra(self, start: Tuple[int, int], finish: Tuple[int, int]) -> float:
        visited = {start: (0, None), }
        unvisited = dict()
        current_cell = start

        while current_cell != finish:
            for dx, dy in [(-1, 0), (1, 0), (0, 1), (0, -1)]:  # N, S, E, W
                neighbour = (current_cell[0] + dx, current_cell[1] + dy)

                # Skip if neighbour is out of the map boundaries or was already visited
                if not 0 <= neighbour[0] < self.h or not 0 <= neighbour[1] < self.w or neighbour in visited.keys():
                    continue

                # Skip neighbours that cannot be accessed
                if self.map[neighbour[0]][neighbour[1]] <= 0:
                    continue

                neighbour_cost = self.map[neighbour[0]][neighbour[1]] + visited[current_cell][0]

                # Store cost to reach neighbour
                stored_neighbour_cost = unvisited.get(neighbour, (None, None))[0]
                if stored_neighbour_cost is None or neighbour_cost < stored_neighbour_cost:
                    unvisited[neighbour] = (neighbour_cost, current_cell)

            if not unvisited:
                raise Exception('No path is feasible!')

            current_cell = min(unvisited, key=unvisited.get)
            visited[current_cell] = unvisited.pop(current_cell)

        path = [finish]
        while path[-1] != start:
            path.append(visited[path[-1]][1])
        path.reverse()

        return visited[finish][0]

    def view_map(self):
        if self.map is None:
            raise AttributeError('World not initialized yet!')

        view_map = self.map
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
