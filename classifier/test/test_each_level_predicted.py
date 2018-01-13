import math
import unittest

import numpy as np
import torch
from torch import FloatTensor
from scipy.sparse import csr_matrix
from torch.autograd import Variable

from classifier import LCPLPredicted
from data.Dataset import Dataset


class TempDoc2vec():

    def transform(self, data):
        indice = [j for i in data for j in i]
        indptr = np.cumsum([0] + [len(i) for i in data])
        data_one = np.ones(len(indice))
        return csr_matrix((data_one, indice, indptr), shape=(len(data), 7)).toarray()


class TestEachLevelPredicted(unittest.TestCase):

    def setUp(self):
        self.model = LCPLPredicted(7, 2, 5, 3, 2, use_dropout=False)
        self.model.dense.weight.data.fill_(1)
        self.model.dense.bias.data.zero_()
        self.model.prev_dense.weight.data.fill_(1)
        self.model.prev_dense.bias.data.zero_()
        self.model.logit.weight.data.fill_(0.2)
        self.model.logit.bias.data.zero_()
        self.dataset = Dataset("test", 1, "train")
        doc2vec = TempDoc2vec()
        self.dataset.change_to_Doc2Vec(doc2vec)

    def test_forward(self):
        real_result = [44.0, 44.0]
        for datas, _ in self.dataset.generate_batch(0, 1):
            prev_target = FloatTensor([[7.0, 7.0]])
            datas = torch.cat([datas, prev_target], 1)
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
        self.model.prev_dense.weight.data.fill_(1)
        self.model.prev_dense.bias.data.zero_()
        self.model.logit.weight.data.fill_(2)
        self.model.logit.bias.data.zero_()
        self.model.eval()
        real_result = [440.0, 440.0]
        for datas, _ in self.dataset.generate_batch(0, 1):
            prev_target = FloatTensor([[7.0, 7.0]])
            datas = torch.cat([datas, prev_target], 1)
            datas = Variable(datas, volatile=True)
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
        real_loss = - math.log(1 / (1 + math.exp(-44)))
        for datas, labels in self.dataset.generate_batch(0, 3):
            prev_target = FloatTensor([[7.0, 7.0], [7.0, 7.0], [7.0, 7.0]])
            datas = torch.cat([datas, prev_target], 1)
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
            prev_target = FloatTensor([[7.0, 7.0], [7.0, 7.0], [7.0, 7.0]])
            datas = torch.cat([datas, prev_target], 1)
            datas = Variable(datas, volatile=True)
            labels = Variable(labels, volatile=True)
            f1_macro, f1_micro = self.model.evaluate(datas, labels)
            self.assertAlmostEqual(real_score, f1_macro, 6)
            self.assertAlmostEqual(real_score, f1_micro, 6)
