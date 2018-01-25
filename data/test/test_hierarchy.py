import unittest

import data.hierarchy as hie


class HierarchyUnitTest(unittest.TestCase):

    def test_create_hierarchy(self):
        hierarchy, parent_of, all_name, name_to_index = hie.create_hierarchy_structure(
            'test/hierarchy.txt')
        real_all_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8']
        real_hierarchy = {0: set([1, 2]),
                          1: set([3, 4]),
                          2: set([5, 7]),
                          3: set([6]),
                          4: set([5]),
                          5: set([6]),
                          6: set([7]),
                          7: set([8])}
        real_parent_of = {1: set([0]),
                          2: set([0]),
                          3: set([1]),
                          4: set([1]),
                          5: set([2, 4]),
                          6: set([3, 5]),
                          7: set([2, 6]),
                          8: set([7])}
        real_name_to_index = {'0': 0,
                              '1': 1,
                              '2': 2,
                              '3': 3,
                              '4': 4,
                              '5': 5,
                              '6': 6,
                              '7': 7,
                              '8': 8}
        self.assertDictEqual(hierarchy, real_hierarchy)
        self.assertDictEqual(parent_of, real_parent_of)
        self.assertDictEqual(name_to_index, real_name_to_index)
        self.assertListEqual(all_name, real_all_name)

    def test_find_first_level(self):
        _, parent_of, all_name, _ = hie.create_hierarchy_structure(
            'test/hierarchy.txt')
        first_level, _ = hie.find_first_level(parent_of, all_name)
        real_first_level = set([0])
        self.assertSetEqual(first_level, real_first_level)

    def test_find_level(self):
        hierarchy, parent_of, all_name, _ = hie.create_hierarchy_structure(
            'test/hierarchy.txt')
        level = hie.find_level(hierarchy, parent_of, all_name)
        real_level = [{1, 2},
                      {3, 4},
                      {5},
                      {6},
                      {7},
                      {8}]
        self.assertListEqual(level, real_level)

    def test_remap_index(self):
        hierarchy, parent_of, all_name, name_to_index, level = hie.reindex_hierarchy(
            'test/hierarchy.txt')
        real_all_name = ['1', '2', '3', '4', '5', '6', '7', '8']
        real_hierarchy = {0: set([2, 3]),
                          1: set([4, 6]),
                          2: set([5]),
                          3: set([4]),
                          4: set([5]),
                          5: set([6]),
                          6: set([7])}
        real_parent_of = {2: set([0]),
                          3: set([0]),
                          4: set([1, 3]),
                          5: set([2, 4]),
                          6: set([1, 5]),
                          7: set([6])}
        real_name_to_index = {'1': 0,
                              '2': 1,
                              '3': 2,
                              '4': 3,
                              '5': 4,
                              '6': 5,
                              '7': 6,
                              '8': 7}
        real_level = [0, 2, 4, 5, 6, 7, 8]
        self.assertSequenceEqual(real_hierarchy, hierarchy)
        self.assertSequenceEqual(real_parent_of, parent_of)
        self.assertSequenceEqual(real_all_name, all_name)
        self.assertSequenceEqual(real_name_to_index, name_to_index)
        self.assertSequenceEqual(real_level, level.tolist())

    def test_load_save_hierarchy(self):
        hierarchy, parent_of, all_name, name_to_index, level = hie.reindex_hierarchy(
            'test/hierarchy.txt')
        hie.save_hierarchy("test/hierarchy.pickle", hierarchy,
                           parent_of, all_name, name_to_index, level)
        real_hierarchy, real_parent_of, real_all_name, real_name_to_index, real_level = hie.load_hierarchy(
            "test/hierarchy.pickle")
        self.assertSequenceEqual(real_hierarchy, hierarchy)
        self.assertSequenceEqual(real_parent_of, parent_of)
        self.assertSequenceEqual(real_all_name, all_name)
        self.assertSequenceEqual(real_name_to_index, name_to_index)
        self.assertSequenceEqual(real_level.tolist(), level.tolist())

    def test_cutoff_index(self):
        hierarchy, parent_of, all_name, name_to_index, level = hie.reindex_hierarchy(
            'test/hierarchy.txt')
        hierarchy, parent_of, all_name, name_to_index, level, _ = hie.cutoff_label(
            [2, 6, 7], hierarchy, parent_of, all_name, name_to_index, level)
        real_all_name = ['1', '2', '4', '5', '6']
        real_hierarchy = {0: set([2]),
                          1: set([3]),
                          2: set([3]),
                          3: set([4])}
        real_parent_of = {2: set([0]),
                          3: set([1, 2]),
                          4: set([3])}
        real_name_to_index = {'1': 0,
                              '2': 1,
                              '4': 2,
                              '5': 3,
                              '6': 4}
        real_level = [0, 2, 3, 4, 5]
        self.assertSequenceEqual(real_hierarchy, hierarchy)
        self.assertSequenceEqual(real_parent_of, parent_of)
        self.assertSequenceEqual(real_all_name, all_name)
        self.assertSequenceEqual(real_name_to_index, name_to_index)
        self.assertSequenceEqual(real_level, level)
