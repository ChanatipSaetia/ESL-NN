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


class EmbeddingDataset(Dataset):

    def __init__(self, data_name, fold_number, mode="train", state="first", sequence=False):
        super(EmbeddingDataset, self).__init__(
            data_name, fold_number=fold_number, mode=mode, state=state, sequence=sequence)

    def map_vocab(self, tok):
        try:
            return self.index_vocab[tok]
        except KeyError:
            return 0

    def change_to_sparse(self, index_vocab=None):
        self.seq_lengths = LongTensor(list(map(len, self.datas)))
        if index_vocab == None:
            vocab = ['<pad>'] + sorted(list(set(flatten(self.datas))))
            self.index_vocab = {v: i for i, v in enumerate(vocab)}
        else:
            self.index_vocab = index_vocab
        self.dictionary_size = len(self.index_vocab)
        self.datas = [[self.map_vocab(tok)
                       for tok in seq]for seq in self.datas]

        indice = [j for i in self.labels for j in i]
        indptr = np.cumsum([0] + [len(i) for i in self.labels])
        data_one = np.ones(len(indice))
        self.state = "embedding"
        self.labels = csr_matrix((data_one, indice, indptr),
                                 shape=(len(self.labels), len(self.all_name))).tocsc()
        return self.index_vocab

    def generate_batch(self, level, batch_size):
        if self.state != "embedding":
            raise NotEmbeddingState
        number = len(self.datas)
        index = np.arange(0, number, batch_size).tolist()
        index.append(number)
        if level == -1:
            label_level = self.labels.tocsr()
        else:
            label_level = self.labels[:, self.level[level]:self.level[level + 1]].tocsr()

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
