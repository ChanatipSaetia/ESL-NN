import unittest

import numpy as np

import data.hierarchy as hie
import data.preparation as prep
from data.Dataset import Dataset
from data.exception import NotEmbeddingState


class DatasetUnitTest(unittest.TestCase):

    def setUp(self):
        self.dataset_train = Dataset("test", 1, "train")
        self.dataset_validate = Dataset("test", 1, "validate")
        self.dataset_test = Dataset("test", 1, "test")

    def test_hierarchy(self):
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
        self.assertSequenceEqual(real_hierarchy, self.dataset_train.hierarchy)
        self.assertSequenceEqual(real_parent_of, self.dataset_train.parent_of)
        self.assertSequenceEqual(real_all_name, self.dataset_train.all_name)
        self.assertSequenceEqual(
            real_name_to_index, self.dataset_train.name_to_index)
        self.assertSequenceEqual(real_level, self.dataset_train.level.tolist())

    def test_load_data(self):
        file_name = "test/test_data.txt"
        datas, labels = prep.import_data(file_name)
        hierarchy_file_name = "test/hierarchy.pickle"
        labels = prep.map_index_of_label(
            hierarchy_file_name, labels)

        train = self.dataset_train.datas
        validate = self.dataset_validate.datas
        test = self.dataset_test.datas
        train_label = self.dataset_train.labels
        validate_label = self.dataset_validate.labels
        test_label = self.dataset_test.labels
        fold_datas = np.concatenate([train, validate, test])
        fold_labels = np.concatenate(
            [train_label, validate_label, test_label])
        self.assertListEqual(sorted(fold_datas.tolist()), sorted(datas))
        a = sorted(map(list, fold_labels.tolist()))
        b = sorted(map(list, labels))
        self.assertListEqual(a, b)

    def test_cant_use_generate_batch(self):
        with self.assertRaises(NotEmbeddingState):
            for _ in self.dataset_train.generate_batch(0, 1):
                pass

    def test_number_of_each_class(self):
        self.assertIsInstance(
            self.dataset_train.check_each_number_of_class(0), int)
        self.assertEqual(2, self.dataset_train.check_each_number_of_class(0))
        self.assertEqual(2, self.dataset_train.check_each_number_of_class(1))
        self.assertEqual(1, self.dataset_train.check_each_number_of_class(5))

    def test_number_of_level(self):
        self.assertEqual(6, self.dataset_train.number_of_level())
