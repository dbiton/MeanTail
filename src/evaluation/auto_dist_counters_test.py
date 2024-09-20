import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
from random import randint

from estimators.auto_dist_counters import AutoDistCounters

class TestAutoDistCountersSingular(unittest.TestCase):

    def setUp(self):
        size = 10
        self.dist_counters = AutoDistCounters(size)       
        

    def test_initial_state(self):
        self.assertEqual(self.dist_counters.query(1), 0, "Initial counter value should be 0")

    def test_update_new_index(self):
        self.dist_counters.update(1, 1)
        self.assertEqual(self.dist_counters.query(1), 1, "Counter value should be updated to 1")

    def test_query_non_existent_index(self):
        self.assertEqual(self.dist_counters.query(99), 0, "Querying a non-existent index should return 0")

class TestAutoDistCountersMultiple(unittest.TestCase):

    def setUp(self):
        self.counters_count = 100
        self.dist_counters = AutoDistCounters(self.counters_count)

    def test_underfilled_tracked_keys(self):
        for i in range(self.counters_count // 2):
          self.dist_counters.update(i, 1)
        for i in range(self.counters_count // 2):
          self.assertEqual(self.dist_counters.query(i), 1, "Counter value should be updated to 1")

    def test_filled_tracked_keys(self):
        for i in range(self.counters_count):
          self.dist_counters.update(i, 1)
        for i in range(self.counters_count):
          self.assertEqual(self.dist_counters.query(i), 1, "Counter value should be updated to 1")
    
    def test_overfilled_tracked_keys(self):
        for i in range(self.counters_count):
          self.dist_counters.update(i, 1)
        for i in range(10):
            self.dist_counters.update(self.counters_count, 1)
            print(i, self.dist_counters.query(self.counters_count))
        self.assertGreaterEqual(self.dist_counters.query(self.counters_count), 1, "New value should replace already tracked values")



if __name__ == '__main__':
    unittest.main()
