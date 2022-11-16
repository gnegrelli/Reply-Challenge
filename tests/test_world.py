import unittest

from classes.world import World


class TestWorld(unittest.TestCase):

    def test_dijkstra(self):

        world = World(3, 3)
        world.add_world_terrain('###HHHTTT')

        world.dijkstra((1, 1), (2, 2))

    def test_dijkstra_1(self):

        world = World(3, 3)
        world.add_world_terrain('TT#~##~~~')

        world.dijkstra((0, 0), (2, 2))

    def test_dijkstra_2(self):

        world = World(3, 3)
        world.add_world_terrain('T#######~')

        with self.assertRaises(Exception):
            world.dijkstra((0, 0), (2, 2))


if __name__ == '__main__':
    unittest.main()
