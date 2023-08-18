""" Tests voxel methods. """


import unittest
import numpy as np
from comp.pixel import VoxelGrid


class TestVoxel(unittest.TestCase):
    def setUp(self):
        # Create a sample VoxelGrid instance for testing
        time = np.array([1, 2, 3])
        frequency = np.array([10, 20, 30])
        n_range = np.array([0, 1, 2])
        gridValue = np.array([
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
            [[19, 20, 21], [22, 23, 24], [25, 26, 27]]
        ])
        self.voxel_grid = VoxelGrid(time, frequency, n_range, gridValue)

    def test_maxNValues(self):
        top_values = self.voxel_grid.maxValues(3)  # Retrieve top 3 values
        
        # Define the expected top values based on the sample grid
        expected_top_values = [
            (3, 30, 2, 27),
            (3, 30, 1, 26),
            (3, 30, 0, 25)
        ]
        
        self.assertEqual(top_values, expected_top_values)

if __name__ == '__main__':
    unittest.main()