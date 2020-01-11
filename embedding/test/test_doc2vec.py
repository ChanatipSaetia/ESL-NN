import unittest

import numpy as np

from data.Dataset import Dataset
from embedding import Doc2Vec


class TestDoc2vec(unittest.TestCase):

    def setUp(self):
        self.dataset_train = Dataset(
            "test", fold_number=1, mode="train", sequence=True)
        self.dataset_validate = Dataset(
            "test", fold_number=1, mode="validate", sequence=True)
        self.doc2vec = Doc2Vec(
            "test", self.dataset_train.number_of_classes(), min_count=1)

    def test_fit(self):
        self.doc2vec.fit(self.dataset_train.datas, self.dataset_train.labels,
                         self.dataset_validate.datas, self.dataset_validate.labels)

    def test_calcurate_similar(self):
        temp_doc2vec = Doc2Vec("test", 2, min_count=1, batch_size=2)
        tag_vector = np.array([[1, 1, 1], [2, 1, 2]])
        datas = np.array([[1, 2, 1], [2, 1, 2], [1, 2, 2]])
        label = np.array([set([0]), set([1]), set([0, 1])])
        _, _, result = temp_doc2vec.calculate_similar(datas, label, tag_vector)
        self.assertAlmostEqual(0.3798891, result, 3)
