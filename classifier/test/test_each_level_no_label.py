import math
import unittest

import numpy as np
import torch
from scipy.sparse import csr_matrix
from torch.autograd import Variable

from classifier import LCPLNoLabel
from data.Dataset import Dataset


class TempDoc2vec():

    def transform(self, data):
        indice = [j for i in data for j in i]
        indptr = np.cumsum([0] + [len(i) for i in data])
        data_one = np.ones(len(indice))
        return csr_matrix((data_one, indice, indptr), shape=(len(data), 7)).toarray()


class TestEachLevel(unittest.TestCase):

    def setUp(self):
        self.model = LCPLNoLabel(7, 5, 2, use_dropout=False)
        self.model.dense.weight.data.fill_(1)
        self.model.dense.bias.data.zero_()
        self.model.logit.weight.data.fill_(0.2)
        self.model.logit.bias.data.zero_()
        self.dataset = Dataset("test", 1, "train")
        doc2vec = TempDoc2vec()
        self.dataset.change_to_Doc2Vec(doc2vec)

    def test_initial_weight(self):
        number_of_data = self.dataset.number_of_data()
        count = self.dataset.number_of_data_in_each_class()
        self.model.initial_weight(number_of_data, count)
        self.assertListEqual(
            [1.0, 1.0, 1.5, 1.5, 1.5, 1.5, 3.0, 10000.0], self.model.pos_weight.numpy().tolist())

        first_index = self.dataset.index_of_level(0)
        first_count = self.dataset.number_of_data_in_each_class()[
            first_index[0]:first_index[1]]
        self.model.initial_weight(number_of_data, first_count)
        self.assertListEqual(
            [1.0, 1.0], self.model.pos_weight.numpy().tolist())

    def test_forward(self):
        real_result = [2.0, 2.0]
        for datas, _ in self.dataset.generate_batch(0, 1):
            datas = Variable(datas, volatile=True)
            result = self.model.forward(datas)
            self.assertListEqual(
                result.data.numpy().tolist()[0], real_result)
            self.assertFalse(result.requires_grad)

    def test_forward_dropout(self):
        torch.manual_seed = 12345
        self.model.use_dropout = True
        self.model.initial_structure()
        self.model.dense.weight.data.fill_(1)
        self.model.dense.bias.data.zero_()
        self.model.logit.weight.data.fill_(2)
        self.model.logit.bias.data.zero_()
        self.model.eval()
        real_result = [20.0, 20.0]
        for datas, _ in self.dataset.generate_batch(0, 1):
            datas = Variable(datas)
            result = self.model.forward(datas)
            self.assertListEqual(
                result.data.numpy().tolist()[0], real_result)
            result = Variable(result.data)
            self.assertFalse(result.requires_grad)

    def test_train_data(self):
        number_of_data = self.dataset.number_of_data()
        first_index = self.dataset.index_of_level(0)
        first_count = self.dataset.number_of_data_in_each_class()[
            first_index[0]:first_index[1]]
        self.model.initial_weight(number_of_data, first_count)
        real_loss = - math.log(1 / (1 + math.exp(-2)))
        for datas, labels in self.dataset.generate_batch(0, 3):
            datas = Variable(datas)
            labels = Variable(labels)
            loss = self.model.train_model(datas, labels)
            self.assertAlmostEqual(real_loss, loss, 5)

    def test_eval_data(self):
        number_of_data = self.dataset.number_of_data()
        first_index = self.dataset.index_of_level(0)
        first_count = self.dataset.number_of_data_in_each_class()[
            first_index[0]:first_index[1]]
        self.model.initial_weight(number_of_data, first_count)
        real_score = 1
        for datas, labels, in self.dataset.generate_batch(0, 3):
            datas = Variable(datas, volatile=True)
            labels = Variable(labels, volatile=True)
            f1_macro, f1_micro = self.model.evaluate(datas, labels)
            self.assertAlmostEqual(real_score, f1_macro, 6)
            self.assertAlmostEqual(real_score, f1_micro, 6)
