import unittest

import numpy as np
from scipy.sparse import csr_matrix

import data.hierarchy as hie
import data.preparation as prep
from data.Dataset import Dataset


class TempDoc2vec():

    def transform(self, data):
        indice = [j for i in data for j in i]
        indptr = np.cumsum([0] + [len(i) for i in data])
        data_one = np.ones(len(indice))
        return csr_matrix((data_one, indice, indptr), shape=(len(data), 7)).toarray()


class DatasetDoc2vecUnitTest(unittest.TestCase):

    def setUp(self):
        self.dataset = Dataset("test", 1, "train")
        doc2vec = TempDoc2vec()
        self.dataset.change_to_Doc2Vec(doc2vec)

    def test_change_to_Doc2Vec(self):
        label = self.dataset.labels.toarray().astype(int).tolist()
        data = self.dataset.datas.tolist()
        real_data = [
            [0, 0, 1, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 1, 0, 0],
        ]
        real_label = [
            [1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0],
        ]
        self.assertListEqual(data, real_data)
        self.assertListEqual(label, real_label)

    def test_generate_batch(self):
        real_label = [[
            [1, 1],
            [1, 1],
            [1, 1],
        ], [
            [1, 1],
            [1, 1],
            [0, 0],
        ]]
        real_data = [
            [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        ]
        for i in range(20):
            for l in range(2):
                label_index = 0
                for data, label in self.dataset.generate_batch(l, 1):
                    self.assertSequenceEqual(
                        label.numpy().reshape(-1).tolist(), real_label[l][label_index])
                    self.assertSequenceEqual(
                        data.numpy().reshape(-1).tolist(), real_data[label_index])
                    label_index = label_index + 1

    def test_number_of_data_in_class(self):
        real_number = [3, 3, 2, 2, 2, 2, 1, 0]
        number = self.dataset.number_of_data_in_each_class().astype(int).tolist()[
            0]
        self.assertListEqual(real_number, number)
