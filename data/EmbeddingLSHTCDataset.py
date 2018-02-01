import itertools
import os
import pickle

import numpy as np
import torch
from scipy.sparse import csr_matrix
from torch import FloatTensor, LongTensor

import data.hierarchy as hie
import data.preparation as prep
from data import Dataset
from data.exception import NotEmbeddingState

from collections import Counter


def flatten(l):
    return list(itertools.chain.from_iterable(l))


class EmbedLSHTCDataset(Dataset):

    def __init__(self, data_name, mode="train", state="first", least_data=5):
        self.least_data = least_data
        super(EmbedLSHTCDataset, self).__init__(
            data_name, fold_number=0, mode=mode, state=state, sequence=False)

    def load_datas(self):
        if self.state == 'embedding':
            with open('data/%s/doc2vec/data.%s.pickle' % (self.data_name, self.mode), mode='rb') as f:
                self.datas, self.labels = pickle.load(f)
            return
        if not os.path.isfile("data/%s/pickle/data.pickle.%s" %
                              (self.data_name, self.mode)):
            prep.split_data_lshtc(self.data_name, least_data=self.least_data)
            self.load_hierarchy()
        self.datas, self.labels = prep.load_data_in_pickle(
            "%s/pickle/data.pickle.%s" % (self.data_name, self.mode))

    def change_to_sparse(self, vocab):
        self.seq_lengths = LongTensor(list(map(len, self.datas)))
        vocab = ['<pad>'] + sorted(list(set(flatten(self.datas))))
        index_vocab = {v: i for i, v in enumerate(vocab)}
        self.dictionary_size = len(index_vocab)
        self.datas = [[index_vocab(tok)
                       for tok in seq]for seq in self.datas]

        indice = [j for i in self.labels for j in i]
        indptr = np.cumsum([0] + [len(i) for i in self.labels])
        data_one = np.ones(len(indice))
        self.state = "embedding"
        self.labels = csr_matrix((data_one, indice, indptr),
                                 shape=(len(self.labels), len(self.all_name))).tocsc()
        return vocab

    def generate_batch(self, level, batch_size):
        if self.state != "embedding":
            raise NotEmbeddingState
        number = len(self.datas)
        index = np.arange(0, number, batch_size).tolist()
        index.append(number)
        if level == -1:
            label_level = self.labels.tocsr()
        else:
            label_level = self.labels[:, self.level[level]                                      :self.level[level + 1]].tocsr()

        for i in range(len(index) - 1):
            start, end = [index[i], index[i + 1]]
            batch_datas = torch.zeros(
                (len(self.datas[start:end]), self.seq_lengths.max())).long()
            for idx, (seq, seqlen) in enumerate(zip(self.datas[start:end], self.seq_lengths[start:end])):
                batch_datas[idx, :seqlen] = LongTensor(seq)
            batch_labels = FloatTensor(label_level[start:end].toarray())
            yield batch_datas, batch_labels

    def size_of_feature(self):
        return self.dictionary_size

    def get_feature_count(self):
        feature_count = Counter([j for i in self.datas for j in i])
        return [0] + [self.number_of_data() / feature_count[i] for i in sorted(feature_count.keys())]
