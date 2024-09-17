import unittest
import numpy as np

from harvest.helpers import HarvestTile, HarvestMap, KERNEL, PROBABILITY_FUNCS
from harvest.game import HarvestGame


class TestHarvestGame(unittest.TestCase):

    def setUp(self):
        # Tests which operate on __init__ should not have a map built for them.
        if self._testMethodName in [
            "test_init_fails_with_non_2d_map", "test_init_fails_with_apples_in_map", 
            "test_init_fails_with_no_spawn_points", "test_init_fails_with_invalid_spawn_points",
            "test_parse_map"
        ]: return

        # Set up a small test map (3x3 grid) with orchards and empty spaces
        self.test_map = np.array([
            [HarvestTile.EMPTY, HarvestTile.ORCHARD, HarvestTile.EMPTY],
            [HarvestTile.ORCHARD, HarvestTile.EMPTY, HarvestTile.ORCHARD],
            [HarvestTile.EMPTY, HarvestTile.ORCHARD, HarvestTile.EMPTY],
            [HarvestTile.EMPTY, HarvestTile.ORCHARD, HarvestTile.EMPTY],
        ], dtype = np.dtype(HarvestTile))
        self.spawn_points = [(0, 0), (0, 2), (2, 0)]
        self.game = HarvestGame(map = self.test_map, spawn_points = self.spawn_points, seed = 42)

    def test_init_fails_with_non_2d_map(self):
        """[HarvestGame] should not be instantiated with a non-2D map."""

        # Empty map should fail.
        with self.assertRaises(AssertionError):
            game = HarvestGame(map = np.array([]), spawn_points = [])

        # 1D map should fail.
        with self.assertRaises(AssertionError):
            game = HarvestGame(map = np.array([1, 2, 3]), spawn_points = [])

        # 3D map should fail.
        with self.assertRaises(AssertionError):
            game = HarvestGame(map = np.zeros((3, 3, 3)), spawn_points = [])

    def test_init_fails_with_apples_in_map(self):
        """[HarvestGame] should not be instantiated with apples in the map."""

        map_with_apples = np.array([
            [HarvestTile.APPLE, HarvestTile.ORCHARD],
            [HarvestTile.ORCHARD, HarvestTile.EMPTY]
        ], dtype = np.dtype(HarvestTile))
        with self.assertRaises(AssertionError):
            game = HarvestGame(map=map_with_apples, spawn_points=[])

    def test_init_fails_with_no_spawn_points(self):
        """[HarvestGame] should not be instantiated with no spawn points."""

        with self.assertRaises(AssertionError):
            game = HarvestGame(map=np.zeros((3, 3)), spawn_points=[])
    
    def test_init_fails_with_invalid_spawn_points(self):
        """[HarvestGame] should not be instantiated with invalid spawn points."""

        with self.assertRaises(AssertionError):
            game = HarvestGame(map=np.zeros((3, 3)), spawn_points=[(-1, -1), (3, 3)])

    def test_getitem(self):
        """[HarvestGame] should return the correct tile at a given coordinate."""

        self.assertEqual(self.game[0, 0], HarvestTile.EMPTY)
        self.assertEqual(self.game[1, 1], HarvestTile.EMPTY)
        self.assertEqual(self.game[0, 1], HarvestTile.ORCHARD)

    def test_setitem(self):
        """[HarvestGame] should update the map correctly when setting a tile."""

        self.assertEqual(self.game[0, 0], HarvestTile.EMPTY)
        self.game[0, 0] = HarvestTile.APPLE
        self.assertEqual(self.game[0, 0], HarvestTile.APPLE)

    def test_iter(self):
        """[HarvestGame] should iterate over the map correctly."""

        expected_tiles = [
            ((0, 0), HarvestTile.EMPTY), ((1, 0), HarvestTile.ORCHARD), ((2, 0), HarvestTile.EMPTY),
            ((0, 1), HarvestTile.ORCHARD), ((1, 1), HarvestTile.EMPTY), ((2, 1), HarvestTile.ORCHARD),
            ((0, 2), HarvestTile.EMPTY), ((1, 2), HarvestTile.ORCHARD), ((2, 2), HarvestTile.EMPTY),
            ((0, 3), HarvestTile.EMPTY), ((1, 3), HarvestTile.ORCHARD), ((2, 3), HarvestTile.EMPTY),
        ]
        result = list(self.game)
        self.assertEqual(result, expected_tiles)

    def test_parse_map(self):
        """[HarvestGame] should convert a list of strings to a HarvestMap correctly."""

        map_strings = [
            ".O.",
            "O.O",
            ".O.",
            ".O."
        ]
        parsed_map = HarvestGame.parse_map(map_strings)
        expected_map = np.array([
            [HarvestTile.EMPTY, HarvestTile.ORCHARD, HarvestTile.EMPTY],
            [HarvestTile.ORCHARD, HarvestTile.EMPTY, HarvestTile.ORCHARD],
            [HarvestTile.EMPTY, HarvestTile.ORCHARD, HarvestTile.EMPTY],
            [HarvestTile.EMPTY, HarvestTile.ORCHARD, HarvestTile.EMPTY],
        ], dtype = np.dtype(HarvestTile))
        np.testing.assert_array_equal(parsed_map, expected_map)

    def test_width(self):
        """[HarvestGame] should return the correct width of the map."""

        self.assertEqual(self.game.width, 3)

    def test_height(self):
        """[HarvestGame] should return the correct height of the map."""

        self.assertEqual(self.game.height, 4)

    def test_in_bounds(self):
        """[HarvestGame] should return whether a coordinate is in bounds correctly."""

        self.assertTrue(self.game.in_bounds((1, 1)))
        self.assertFalse(self.game.in_bounds((-1, 1)))
        self.assertFalse(self.game.in_bounds((3, 3)))

    def test_setup(self):
        """[HarvestGame] should spawn apples in orchards correctly."""

        self.game.setup(2)
        apple_count = np.sum(self.game.map == HarvestTile.APPLE.value)
        self.assertEqual(apple_count, 2)

    def test_view_map(self):
        """[HarvestGame] should return the correct string representation of the map."""

        expected_map_view = ".O.\nO.O\n.O.\n.O."
        self.assertEqual(self.game.view_map(), expected_map_view)

    def test_view_partial_map(self):
        """[HarvestGame] should return the correct string representation of a partial map."""

        center = (1, 1)
        radius = 1
        expected_partial_view = ".O.\nO.O\n.O."
        self.assertEqual(self.game.view_partial_map(center, radius), expected_partial_view)

        radius = 0
        expected_partial_view = "."
        self.assertEqual(self.game.view_partial_map(center, radius), expected_partial_view)

        radius = 2
        expected_partial_view = ".O.\nO.O\n.O.\n.O."
        self.assertEqual(self.game.view_partial_map(center, radius), expected_partial_view)


if __name__ == '__main__':
    unittest.main()
