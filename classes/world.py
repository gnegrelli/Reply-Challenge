import re

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from typing import List, Set, Tuple


class World(object):
    """Class to create world map"""

    MAP_VALUES = {
        '~': 800,
        '*': 200,
        '+': 150,
        'X': 120,
        '_': 100,
        'H': 70,
        'T': 50,
    }

    def __init__(self, width: int, height: int, **kwargs) -> None:
        # Validate inputs
        assert height > 0 and isinstance(height, int), 'Value for map height must be a positive integer.'
        assert width > 0 and isinstance(width, int), 'Value for map width must be a positive integer.'

        self.w = width
        self.h = height
        self.map = None

    def add_world_terrain(self, string_terrain: str) -> None:
        """Method to add cost to world positions using given string"""
        assert isinstance(string_terrain, str), 'Input must be a string'

        # Remove whitespaces from input string
        clean_string = re.sub(r'\s+', '', string_terrain)
        assert len(clean_string) == self.w*self.h, f'Input string must have a length of {self.w*self.h} chars'

        # Convert input string to cost values
        map_values = [self.MAP_VALUES.get(char, -1) for char in clean_string]

        # Create matrix with cost of each world position
        self.map = np.array(map_values).reshape((self.h, self.w))

    def dijkstra(self, start: Tuple[int, int], finish: Tuple[int, int]) -> Tuple[float, List[Tuple[int, int]]]:
        """Method to find the cheapest path between two points in the world using Dijkstra algorithm"""
        assert 0 <= start[0] <= self.w and 0 <= start[1] <= self.h, 'Invalid coordinate for starting point.'
        assert 0 <= finish[0] <= self.w and 0 <= finish[1] <= self.h, 'Invalid coordinate for finishing point.'

        # Initialize dict of visited and unvisited locations. These dicts store the cell coordinates as key and a
        # tuple as value containing the cost of reaching to the position and from which cell.
        visited = {start: (0, None), }
        unvisited = dict()
        current_location = start

        while current_location != finish:
            # Look for northern, southern, eastern and western neighbours of current location
            for dx, dy in [(-1, 0), (1, 0), (0, 1), (0, -1)]:  # N, S, E, W
                neighbour = (current_location[0] + dx, current_location[1] + dy)

                # Skip if neighbour is out of the map boundaries or was already visited
                if not 0 <= neighbour[0] < self.w or not 0 <= neighbour[1] < self.h or neighbour in visited.keys():
                    continue

                # Skip neighbours that cannot be accessed (value = -1)
                if self.map[neighbour[1]][neighbour[0]] <= 0:
                    continue

                # Calculate cost to reach neighbour from current position
                neighbour_cost = self.map[neighbour[1]][neighbour[0]] + visited[current_location][0]

                # Update cost to reach unvisited neighbour
                stored_neighbour_cost = unvisited.get(neighbour, (None, None))[0]
                if stored_neighbour_cost is None or neighbour_cost < stored_neighbour_cost:
                    unvisited[neighbour] = (neighbour_cost, current_location)

            # Raise exception if there is no feasible path between points
            if not unvisited:
                raise Exception('No path is feasible!')

            # Find the cheapest cell in the list of unvisited and move into it
            current_location = min(unvisited, key=unvisited.get)
            visited[current_location] = unvisited.pop(current_location)

        # Construct path from finishing to starting point and reverse it
        path = [finish]
        while path[-1] != start:
            path.append(visited[path[-1]][1])
        path.reverse()

        return visited[finish][0], path

    def view_map(self) -> None:
        """Plot map using seaborn's heatmap"""
        if self.map is None:
            raise AttributeError('World not initialized yet!')

        sns.heatmap(self.map)
        plt.show()

    def allowed_spots(self, occupied_spots: List[Tuple[int, int]] = None) -> Set[Tuple[int, int]]:
        allowed_spots = {(x, y) for x in range(self.w) for y in range(self.h) if self.map[y][x] >= 0}

        if occupied_spots is None:
            occupied_spots = set()
        else:
            occupied_spots = set(occupied_spots)

        return allowed_spots - occupied_spots

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
