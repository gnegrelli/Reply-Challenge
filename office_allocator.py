from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict


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


if __name__ == '__main__':
    parser = ArgumentParser(description='Script to allocate offices on a map.')
    parser.add_argument('map_file', help='File containing map of terrain')

    args = validate_args(parser)
