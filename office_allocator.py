import random

from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, List

from classes.world import World
from classes.office import Office
from exceptions.world_exceptions import ImpossiblePathException


def validate_args(args: ArgumentParser) -> Dict[str, Any]:
    """Method to validate input arguments"""
    args = args.parse_args()

    # Validate map_file input
    map_file = Path(args.map_file)
    if not map_file.exists():
        raise ValueError(f'File \'{map_file.absolute()}\' does not exist.')

    if map_file.suffix not in ('.txt',):
        raise ValueError(f'File format \'{map_file.suffix}\' not supported.')

    return {'map_file': map_file, }


def read_map_file(file: Path) -> Dict[str, Any]:
    with open(file, 'r') as file:
        map_width, map_height, n_customers, n_offices = map(int, file.readline().strip().split(' '))

        world = World(map_width, map_height)

        customers = []
        for _ in range(n_customers):
            customer_x, customer_y, customer_reward = map(int, file.readline().strip().split(' '))
            customers.append(Office(customer_x, customer_y, customer_reward))

        world.add_world_terrain(file.read())

    return {'world': world, 'customers': customers, 'n_offices': n_offices, }


def allocate_offices(world: World, customers: List[Office], n_offices: int = 1, max_it: int = 10, n_init : int = 5,
                     **kwargs) -> List[Office]:
    """Method based on K-Means clustering to allocate offices in optimal positions"""
    # Create dictionary to store solutions of each iteration
    solutions = dict()

    # Create list of spots offices cannot occupy
    occupied_spots = [customer.location for customer in customers]

    # Since the convergence of K-Means clustering algorithm depends on the initial centroids chosen, we must execute
    # it a few times and to find the optimal solution.
    # Execute the K-Means clustering algorithm for n_init times and store the results
    for init in range(n_init):
        # Randomly allocate the initial centroids (offices)
        offices = [Office(*spot) for spot in random.choices(tuple(world.allowed_spots(occupied_spots)), k=n_offices)]

        for it in range(max_it):
            total_cost = 0
            customer_to_office = dict()

            # Assign each customer to its nearest office
            for customer in customers:
                min_cost = None
                for office in offices:
                    # Retrieve cost between office and customer
                    try:
                        cost, _ = world.calculate_path_cost(office.location, customer.location)
                    except ImpossiblePathException:
                        continue

                    # Store office that is closest to current customer
                    if min_cost is None or cost < min_cost:
                        min_cost = cost
                        customer_to_office[customer] = office

                # Add cost of path to total_cost of solution
                total_cost += min_cost

            # Reevaluate position of offices based on their clusters
            centroids = set()
            for old_centroid in customer_to_office.values():
                # Group customer allocated to same centroid
                cluster = [
                    customer.location for customer, centroid in customer_to_office.items() if centroid == old_centroid
                ]

                # Calculate position of cluster centroid (using average position of points in cluster)
                centroid = tuple(map(sum, zip(*cluster)))
                centroid = tuple(map(lambda p: round(p/len(cluster)), centroid))
                centroids.add(centroid)

            # Check for changes in the location of offices
            if centroids == {office.location for office in offices}:
                # If centroids did not change between iterations, store solution and cost
                solution = tuple(Office(*spot) for spot in centroids)
                solutions[solution] = total_cost
                break
            else:
                # Update offices
                offices = [Office(*centroid) for centroid in centroids]
                # Assure the number of cluster is constant
                if len(offices) < n_offices:
                    offices.extend([
                        Office(*spot) for spot in random.choices(
                            tuple(world.allowed_spots(occupied_spots)), k=n_offices - len(offices)
                        )
                    ])

    # Return solution with the smallest cost
    return list(min(solutions, key=solutions.get))


if __name__ == '__main__':
    parser = ArgumentParser(description='Script to allocate offices on a map.')
    parser.add_argument('map_file', help='File containing map of terrain')

    args = validate_args(parser)
    world_objects = read_map_file(args['map_file'])

    final_offices = allocate_offices(**world_objects)

    world_objects['world'].view_map(world_objects['customers'], final_offices)
