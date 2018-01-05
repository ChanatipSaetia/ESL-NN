import os

import numpy as np
from scipy.sparse import csr_matrix
from torch import FloatTensor, LongTensor

import data.hierarchy as hie
import data.preparation as prep
from data.exception import NotEmbeddingState


class Dataset():

    def __init__(self, data_name, fold_number=1, mode="train", state="first"):
        self.data_name = data_name
        self.fold_number = fold_number
        self.mode = mode
        self.data_type = "index"
        self.state = state
        self.load_hierarchy()
        self.load_datas()
        # sparse data

    def load_hierarchy(self):
        if not os.path.isfile("%s/hierarchy.pickle" % self.data_name):
            hierarchy, parent_of, all_name, name_to_index, level = hie.reindex_hierarchy(
                'test/hierarchy.txt')
            hie.save_hierarchy("test/hierarchy.pickle", hierarchy,
                               parent_of, all_name, name_to_index, level)
        self.hierarchy, self.parent_of, self.all_name, self.name_to_index, self.level = hie.load_hierarchy(
            "%s/hierarchy.pickle" % self.data_name)

    def load_datas(self):
        if not os.path.isfile("%s/fold/data_%d.pickle.%s" %
                              (self.data_name, self.fold_number, self.mode)):
            file_name = "test/test_data.txt"
            datas, labels = prep.import_data(file_name)
            hierarchy_file_name = "test/hierarchy.pickle"
            new_labels = prep.map_index_of_label(
                hierarchy_file_name, labels)
            data_name = "test"
            prep.split_data(datas, new_labels, data_name)
        self.datas, self.labels = prep.load_data_in_pickle(
            "%s/fold/data_%d.pickle.%s" % (self.data_name, self.fold_number, self.mode))

    def number_of_level(self):
        return len(self.level) - 1

    def number_of_classes(self):
        return len(self.all_name)

    def check_each_number_of_class(self, level):
        return self.level[level + 1] - self.level[level]

    def change_to_Doc2Vec(self, doc2vec):
        self.datas = doc2vec.transform(self.datas)

        indice = [j for i in self.labels for j in i]
        indptr = np.cumsum([0] + [len(i) for i in self.labels])
        data_one = np.ones(len(indice))
        self.state = "embedding"
        self.labels = csr_matrix((data_one, indice, indptr),
                                 shape=(len(self.labels), len(self.all_name))).tocsc()

    def generate_batch(self, level, batch_size):
        if self.state != "embedding":
            raise NotEmbeddingState
        number = len(self.datas)
        index = np.arange(0, number, batch_size).tolist()
        index.append(number)
        label_level = self.labels[:, self.level[level]
            :self.level[level + 1]].tocsr()
        for i in range(len(index) - 1):
            start, end = [index[i], index[i + 1]]
            batch_datas = FloatTensor(self.datas[start:end])
            batch_labels = LongTensor(label_level[start:end].toarray())
            yield batch_datas, batch_labels

    def number_of_data_in_each_class(self):
        if self.state != "embedding":
            raise NotEmbeddingState
        return np.sum(self.labels, 0).astype(int)
