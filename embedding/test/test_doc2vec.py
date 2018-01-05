import unittest

from data.Dataset import Dataset
from embedding import Doc2Vec


class TestDoc2vec(unittest.TestCase):

    def setUp(self):
        self.dataset_train = Dataset("test", fold_number=1, mode="train")
        self.dataset_validate = Dataset("test", fold_number=1, mode="validate")
        self.doc2vec = Doc2Vec(
            self.dataset_train.number_of_classes(), min_count=1)

    def test_fit(self):
        self.doc2vec.fit(self.dataset_train.datas, self.dataset_train.labels,
                         self.dataset_validate.datas, self.dataset_validate.labels)

    def test_calcurate_similar(self):
        pass
