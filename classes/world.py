import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from tqdm import tqdm
from typing import List, Set, Tuple, Union

from classes.office import Office
from exceptions.world_exceptions import ImpossiblePathException, UnmappedWorldException


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

    PATHS_FROM_TO = dict()

    def __init__(self, width: int, height: int, **kwargs) -> None:
        # Validate inputs
        assert height > 0 and isinstance(height, int), 'Value for map height must be a positive integer.'
        assert width > 0 and isinstance(width, int), 'Value for map width must be a positive integer.'

        self.w = width
        self.h = height
        self._map = None
        self.costs = None

    def loc_value(self, location: Union[Tuple[int, int], Office]) -> Union[int, None]:
        if self._map is None:
            return None

        if isinstance(location, Office):
            location = location.location

        return self._map[location[1]][location[0]]

    def add_world_terrain(self, string_terrain: str) -> None:
        """Method to add cost to world positions using given string"""
        assert isinstance(string_terrain, str), 'Input must be a string'

        # Remove whitespaces from input string
        clean_string = re.sub(r'\s+', '', string_terrain)
        assert len(clean_string) == self.w * self.h, f'Input string must have a length of {self.w * self.h} chars'

        # Convert input string to cost values
        map_values = [self.MAP_VALUES.get(char, -1) for char in clean_string]

        # Create matrix with cost of each world position
        self._map = np.array(map_values).reshape((self.h, self.w))

    # TODO: This is taking too long. There must be a way to speed it up.
    def calculate_costs_to_customers(self, customers: List[Office]):
        """Method to calculate cost of paths leading to customers"""
        spots = list(self.allowed_spots(customers))
        customer_spots = [customer.location for customer in customers]
        self.costs = pd.DataFrame(np.nan, index=spots, columns=customer_spots)

        for start in tqdm(spots):
            # Calculate cost from start to customers
            for finish in customer_spots:

                # Skip if cost was already calculated
                if self.PATHS_FROM_TO.get((start, finish)) is not None:
                    continue

                try:
                    cost, path = self.calculate_path_cost(start, finish)
                except ImpossiblePathException:
                    # If path is not possible, set it as an empty list
                    self.PATHS_FROM_TO[(start, finish)] = []
                    continue

                # Calculate cost of inner paths
                for i, start_path in enumerate(path[:-1]):
                    # If cost from start_path was already calculated (and consequently all following start_paths),
                    # evaluate path for next customer
                    if self.PATHS_FROM_TO.get((start_path, finish)) is not None:
                        break

                    # Skip calculation of paths starting in a customer spot
                    if start_path in customer_spots:
                        continue

                    # Adjust cost for start points in inner path
                    if start_path != start:
                        cost -= self.loc_value(start_path)

                    # Store cost and path
                    self.costs[finish][start_path] = cost
                    self.PATHS_FROM_TO[(start_path, finish)] = path[i:]

    # TODO: Use self.costs to speed up this calculation (if optimal cost of current location to finish is already
    #  evaluated it is possible to join the solutions and skip the rest of the calculation). To do so, it is probably
    #  necessary to store all the paths in a dataframe.
    #  Not sure if that is really true though. During exploration A* may land in one cell that is currently the
    #  best path, but may not be part of the best route
    def calculate_path_cost(
            self, start: Tuple[int, int], finish: Tuple[int, int]
    ) -> Tuple[float, List[Tuple[int, int]]]:
        """Method to find the cheapest path between two points in the world"""
        assert 0 <= start[0] <= self.w and 0 <= start[1] <= self.h, 'Invalid coordinate for starting point.'
        assert 0 <= finish[0] <= self.w and 0 <= finish[1] <= self.h, 'Invalid coordinate for finishing point.'

        if self.loc_value(start) <= 0:
            raise ImpossiblePathException(f'Cannot create path that starts in position {start}.')

        if self.loc_value(finish) <= 0:
            raise ImpossiblePathException(f'Cannot create path that ends in position {start}.')

        # Check if path was already evaluated
        if self.PATHS_FROM_TO.get((start, finish)) is not None:
            path = self.PATHS_FROM_TO[(start, finish)]
            if not path:
                raise ImpossiblePathException('No path is feasible!')
            return self.costs[finish][start], path

        # Initialize dict of visited and unvisited locations. These dicts store the cell coordinates as key and a
        # tuple as value containing the cost of reaching to the position and from which cell.
        visited = {start: (0, None), }
        unvisited = dict()
        current_location = start

        while current_location != finish:
            # Look for northern, southern, eastern and western neighbours of current location
            for dx, dy in [(-1, 0), (1, 0), (0, 1), (0, -1)]:  # W, E, S, N
                neighbour = (current_location[0] + dx, current_location[1] + dy)

                # Skip if neighbour is out of the map boundaries or was already visited
                if not 0 <= neighbour[0] < self.w or not 0 <= neighbour[1] < self.h or neighbour in visited.keys():
                    continue

                # Skip neighbours that cannot be accessed (value = -1)
                if self.loc_value(neighbour) <= 0:
                    continue

                # Calculate cost to reach neighbour from current position
                neighbour_cost = self.loc_value(neighbour) + visited[current_location][0]

                # Add distance to goal in cost to optimize search
                # TODO: Reevaluate the weight given to distance. Maybe have it as a function argument
                distance_factor = 10 * (abs(finish[0] - neighbour[0]) + abs(finish[1] - neighbour[1]))

                # Update cost to reach unvisited neighbour
                stored_neighbour_cost = unvisited.get(neighbour, (None, None))[0]
                if stored_neighbour_cost is None or neighbour_cost + distance_factor < stored_neighbour_cost:
                    unvisited[neighbour] = (neighbour_cost, current_location)

            # Raise exception if there is no feasible path between points
            if not unvisited:
                raise ImpossiblePathException('No path is feasible!')

            # Find the cheapest cell in the list of unvisited and move into it
            current_location = min(unvisited, key=unvisited.get)
            visited[current_location] = unvisited.pop(current_location)

        # Construct path from finishing to starting point and reverse it
        path = [finish]
        cost = 0
        while path[-1] != start:
            cost += self.loc_value(path[-1])
            path.append(visited[path[-1]][1])
        path.reverse()

        return cost, path

    def view_map(self, customers: List[Office] = None, offices: List[Office] = None,
                 paths: List[List[Tuple[int, int]]] = None, **kwargs) -> None:
        """Plot map using seaborn's heatmap"""
        if self._map is None:
            raise UnmappedWorldException('World not initialized yet!')

        bins = np.array(list(reversed(self.MAP_VALUES.values())))
        discrete_map = np.array([np.digitize(row, bins) for row in self._map])

        ax = sns.heatmap(discrete_map, linewidths=.5, square=True, cmap=sns.cubehelix_palette(len(bins)), **kwargs)

        if paths is not None:
            for path in map(np.array, paths):
                path = path + .5 * np.ones_like(path)
                x, y = zip(*path)
                ax.plot(x, y, '-.')

        if customers is not None:
            for customer in customers:
                ax.plot(*map(lambda loc: loc + .5, customer.location), 'go')

        if offices is not None:
            for office in offices:
                ax.plot(*map(lambda loc: loc + .5, office.location), 'y*')

        plt.show()

    def allowed_spots(self, occupied_spots: List[Union[Office, Tuple[int, int]]] = None) -> Set[Tuple[int, int]]:
        allowed_spots = {(x, y) for x in range(self.w) for y in range(self.h) if self.loc_value((x, y)) >= 0}

        if occupied_spots is None:
            occupied_spots = set()
        else:
            occupied_spots = {spot.location if isinstance(spot, Office) else spot for spot in occupied_spots}

        return allowed_spots - occupied_spots

    def __str__(self) -> str:
        if self._map is None:
            return 'World not initialized yet!'

        string = ''
        for row in self._map:
            string += '|'
            for item in row:
                string += f'{item:^6}|'
            string += '\n'
        return string
