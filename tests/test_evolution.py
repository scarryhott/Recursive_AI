import unittest
import os
os.environ["SKIP_SELF_TESTS"] = "1"
from recursive_ai import RecursiveAI

class TestEvolution(unittest.TestCase):
    def test_evolve_bounds(self):
        ai = RecursiveAI()
        new_ai = ai.evolve()
        self.assertGreaterEqual(new_ai.truth_score, 0.0)
        self.assertLessEqual(new_ai.truth_score, 1.0)

if __name__ == "__main__":
    unittest.main()
