import unittest

from dist_counters import DistCounters

class TestDistCountersSingular(unittest.TestCase):

    def setUp(self):
        distribution = lambda index : 1
        size = 10
        self.dist_counters = DistCounters(size, distribution)

    def test_initial_state(self):
        self.assertEqual(self.dist_counters.query(1), 0, "Initial counter value should be 0")

    def test_update_new_index(self):
        self.dist_counters.update(1, 5)
        self.assertEqual(self.dist_counters.query(1), 5, "Counter value should be updated to 5")

    def test_query_non_existent_index(self):
        self.assertEqual(self.dist_counters.query(99), 0, "Querying a non-existent index should return 0")

    def test_update_existing_index(self):
      self.dist_counters.update(1, 5)
      self.dist_counters.update(1, 3)
      self.assertEqual(self.dist_counters.query(1), 8, "Counter value should be updated to 8 after second update")

    def test_update_negative_value(self):
        self.dist_counters.update(1, 10)
        self.dist_counters.update(1, -4)
        self.assertEqual(self.dist_counters.query(1), 6, "Counter value should be updated to 6 after adding -4")


if __name__ == '__main__':
    unittest.main()
