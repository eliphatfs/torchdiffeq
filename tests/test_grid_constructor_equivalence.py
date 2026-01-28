import torch
import unittest
from torchdiffeq._impl.solvers import FixedGridODESolver

class TestGridConstructorEquivalence(unittest.TestCase):
    def test_grid_constructor_equivalence(self):
        # We want to test that the logic in _grid_constructor_from_step_size
        # produces the same result as the previous version.
        # Since we changed it in place, we will test the current version
        # against our known expected behavior.
        
        step_size = torch.tensor(0.3)
        t = torch.tensor([0.0, 1.0])
        
        # Manually compute what the old version would have done:
        # niters = ceil((1.0 - 0.0) / 0.3 + 1) = ceil(3.333 + 1) = ceil(4.333) = 5
        # t_infer = [0.0, 0.3, 0.6, 0.9, 1.2]
        # t_infer[-1] = 1.0  => [0.0, 0.3, 0.6, 0.9, 1.0]
        
        expected_grid = torch.tensor([0.0, 0.3, 0.6, 0.9, 1.0])
        
        grid_constructor = FixedGridODESolver._grid_constructor_from_step_size(step_size)
        actual_grid = grid_constructor(None, None, t)
        
        torch.testing.assert_close(actual_grid, expected_grid)

    def test_grid_constructor_equivalence_complex(self):
        step_size = torch.tensor(0.01)
        t = torch.tensor([1.5, 2.1])
        
        # Expected:
        # start = 1.5, end = 2.1
        # niters = ceil((2.1 - 1.5) / 0.01 + 1) = ceil(0.6 / 0.01 + 1) = ceil(60 + 1) = 61
        # t_infer = arange(0, 61) * 0.01 + 1.5
        # t_infer[60] = 60 * 0.01 + 1.5 = 0.6 + 1.5 = 2.1
        # Then t_infer[-1] = 2.1 (no change in this specific case, but test covers the flow)
        
        grid_constructor = FixedGridODESolver._grid_constructor_from_step_size(step_size)
        actual_grid = grid_constructor(None, None, t)
        
        self.assertEqual(len(actual_grid), 61)
        self.assertEqual(actual_grid[0], 1.5)
        self.assertEqual(actual_grid[-1], 2.1)
        
        # Check an intermediate point
        self.assertAlmostEqual(actual_grid[10].item(), 1.6)

if __name__ == '__main__':
    unittest.main()
