from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict

from classes.world import World
from classes.office import Office


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


if __name__ == '__main__':
    parser = ArgumentParser(description='Script to allocate offices on a map.')
    parser.add_argument('map_file', help='File containing map of terrain')

    args = validate_args(parser)
    read_map_file(args['map_file'])
