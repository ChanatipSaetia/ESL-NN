import unittest

import numpy as np
from scipy.sparse import csr_matrix
from torch import FloatTensor, ByteTensor
from torch.autograd import Variable

from assemble_classifier import AssembleNoLabel
from data.Dataset import Dataset


class TempDoc2vec():

    def transform(self, data):
        indice = [j for i in data for j in i]
        indptr = np.cumsum([0] + [len(i) for i in data])
        data_one = np.ones(len(indice))
        return csr_matrix((data_one, indice, indptr), shape=(len(data), 7)).toarray()


class TestAssembleNoLabel(unittest.TestCase):

    def setUp(self):
        self.dataset = Dataset("test", 1, "train")
        self.dataset_validate = Dataset("test", 1, "validate")
        self.dataset_test = Dataset("test", 1, "test")
        doc2vec = TempDoc2vec()
        self.dataset.change_to_Doc2Vec(doc2vec)
        self.dataset_validate.change_to_Doc2Vec(doc2vec)
        self.dataset_test.change_to_Doc2Vec(doc2vec)
        hidden = [5] * self.dataset.number_of_level()
        self.model = AssembleNoLabel(
            "test", self.dataset, self.dataset_validate, self.dataset_test, 30, 3, hidden, stopping_time=3)
        self.model.classifier[0].dense.weight.data.fill_(1)
        self.model.classifier[0].dense.bias.data.zero_()
        self.model.classifier[0].logit.weight.data.fill_(0.2)
        self.model.classifier[0].logit.bias.data.zero_()

    def test_initial_model(self):
        for i in range(self.dataset.number_of_level()):
            test_model = self.model.classifier[i]
            number_of_class = self.dataset.check_each_number_of_class(i)
            level = self.dataset.index_of_level(i)
            self.assertEqual(test_model.input_size, 7)
            self.assertEqual(test_model.hidden_size, 5)
            self.assertEqual(test_model.number_of_class, number_of_class)
            self.assertEqual(test_model.level, level)

    def test_score_each_level(self):
        f1_macro, f1_micro = self.model.evaluate_each_level(0, "train")
        real_score = 1
        self.assertAlmostEqual(real_score, f1_macro, 6)
        self.assertAlmostEqual(real_score, f1_micro, 6)

    def test_evaluate(self):
        f1_macro, f1_micro, f1_each = self.model.evaluate("train")
        real_score = [1, 4 / 5, 4 / 5, 4 / 5, 1 / 2, 0]
        self.assertAlmostEqual(0.7125, f1_macro, 6)
        for f1, real in zip(f1_each, real_score):
            self.assertAlmostEqual(real, f1[0], 6)
            # self.assertAlmostEqual(real, f1[1], 6)

    def test_train(self):
        # just train successfully
        self.model.train()
        f1_macro, f1_micro = self.model.evaluate_each_level(0, "train")
        real_score = 1
        self.assertAlmostEqual(real_score, f1_macro, 6)
        self.assertAlmostEqual(real_score, f1_micro, 6)

    def test_threshold_tuning(self):
        self.model.train()
        self.model.tuning_threshold()
        f1_macro, f1_micro = self.model.evaluate_each_level(0, "train")
        real_score = 1
        self.assertAlmostEqual(real_score, f1_macro, 6)
        self.assertAlmostEqual(real_score, f1_micro, 6)

    def test_correction(self):
        test_label = [[0, 0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0, 1, 0, 0]]
        real_result_label = [[1, 0, 0, 1, 0, 0, 0, 0],
                             [1, 1, 1, 1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1, 1, 0, 0]]
        torch_label = ByteTensor(test_label)
        result = self.model.child_based_correction(
            torch_label).cpu().numpy().tolist()
        for label, real_label in zip(result, real_result_label):
            self.assertListEqual(
                real_label, label)
