import os
import pickle
from functools import reduce

import numpy as np
from scipy.sparse import csr_matrix
from torch import FloatTensor

import data.hierarchy as hie
import data.preparation as prep
from data.exception import NotEmbeddingState


class Dataset():

    def __init__(self, data_name, mode="train", fold_number=0, state="first", sequence=True, test_split=True, classification_type="hierarchical", data_file_name="data.txt"):
        self.data_name = data_name
        self.fold_number = fold_number
        self.mode = mode
        self.data_type = "index"
        self.state = state
        self.sequence = sequence
        self.test_split = test_split
        self.classification_type = classification_type
        self.data_file_name = data_file_name
        self.load_hierarchy()
        self.load_datas()

    def load_hierarchy(self):
        if not os.path.isfile("data/%s/hierarchy.pickle" % self.data_name):
            if self.classification_type == 'hierarchical':
                hierarchy, parent_of, all_name, name_to_index, level = hie.reindex_hierarchy(
                    '%s/hierarchy.txt' % self.data_name)
                hie.save_hierarchy("%s/hierarchy.pickle" % self.data_name, hierarchy,
                                   parent_of, all_name, name_to_index, level)
            else:
                return
        self.hierarchy, self.parent_of, self.all_name, self.name_to_index, self.level = hie.load_hierarchy(
            "%s/hierarchy.pickle" % self.data_name)
        self.not_leaf_node = np.array([i in list(
            self.hierarchy.keys()) for i in range(self.number_of_classes())])

    def create_temp_hierarchy(self, all_name):
        if self.classification_type != 'hierarchical':
            name_to_index = {}
            for i, n in enumerate(all_name):
                name_to_index[n] = i
            hie.save_hierarchy("%s/hierarchy.pickle" % self.data_name, {},
                               {}, all_name, name_to_index, [0, len(all_name)])
            self.hierarchy, self.parent_of, self.all_name, self.name_to_index, self.level = hie.load_hierarchy(
                "%s/hierarchy.pickle" % self.data_name)
            self.not_leaf_node = np.array([i in list(
                self.hierarchy.keys()) for i in range(self.number_of_classes())])

    def load_datas(self):
        if self.state == 'embedding':
            with open('data/%s/doc2vec/data.%s.pickle' % (self.data_name, self.mode), mode='rb') as f:
                self.datas, self.labels = pickle.load(f)
            self.create_label_stat()
            return
        if self.fold_number == 0:
            store_name = "%s/store/data.pickle.%s" % (
                self.data_name, self.mode)
        else:
            store_name = "%s/fold/data_%d.pickle.%s" % (
                self.data_name, self.fold_number, self.mode)

        if not os.path.isfile("data/" + store_name):
            if not self.test_split:
                file_name = "%s/%s" % (self.data_name,
                                       self.data_file_name)
            else:
                file_name = "%s/%s" % (self.data_name,
                                       self.data_file_name)
            datas, labels = prep.import_data(file_name, sequence=self.sequence)
            self.create_temp_hierarchy(
                list(set([j for i in labels for j in i])))
            hierarchy_file_name = "%s/hierarchy.pickle" % self.data_name
            new_labels = prep.map_index_of_label(
                hierarchy_file_name, labels)
            if not self.test_split:
                if self.mode == "train":
                    prep.split_validate_data(
                        datas, new_labels, self.data_name)
                else:
                    directory = "data/%s/store" % self.data_name
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    with open('data/' + store_name, 'wb') as f:
                        pickle.dump([datas, new_labels], f)
                        f.close()
            else:
                if self.fold_number == 0:
                    prep.split_data(datas, new_labels, self.data_name)
                else:
                    prep.split_fold_data(datas, new_labels, self.data_name)
        self.datas, self.labels = prep.load_data_in_pickle(store_name)

    def number_of_level(self):
        return len(self.level) - 1

    def number_of_classes(self):
        return self.level[-1]

    def number_of_parent_classes(self, level):
        return int(np.sum(self.filter_parent(level)))

    def filter_parent(self, level):
        return self.not_leaf_node[self.level[level]:self.level[level + 1]]

    def check_each_number_of_class(self, level):
        return int(self.level[level + 1] - self.level[level])

    def change_to_Doc2Vec(self, doc2vec):
        self.datas = doc2vec.transform(self.datas)

        indice = [j for i in self.labels for j in i]
        indptr = np.cumsum([0] + [len(i) for i in self.labels])
        data_one = np.ones(len(indice))
        self.state = "embedding"
        self.labels = csr_matrix((data_one, indice, indptr),
                                 shape=(len(self.labels), len(self.all_name))).tocsc()
        if not os.path.exists('export/%s/doc2vec/' % self.data_name):
            os.makedirs('export/%s/doc2vec/' % self.data_name)
        with open('export/%s/doc2vec/%s.txt' % (self.data_name, self.mode), mode='w') as f:
            for i in self.datas:
                f.write(' '.join(map(str, i)) + '\n')
        self.create_label_stat()

    def delete_label_more(self, n):
        new_labels = []
        for i in self.labels:
            s = set()
            for j in i:
                if j < n:
                    s.add(j)
            new_labels.append(s)
        self.labels = new_labels

    def create_label_stat(self):
        sum_label = np.sum(
            self.labels[:, np.invert(self.not_leaf_node)], 1)
        self.max_label = int(np.max(sum_label))
        self.min_label = int(np.min(sum_label))
        # sum_label = np.log(sum_label)
        self.mean_label = np.mean(sum_label)
        self.sd_label = np.std(sum_label)

    def generate_batch(self, level, batch_size):
        if self.state != "embedding":
            raise NotEmbeddingState
        number = len(self.datas)
        index = np.arange(0, number, batch_size).tolist()
        index.append(number)
        if level == -1:
            label_level = self.labels.tocsr()
        else:
            label_level = self.labels[:, self.level[level]
                :self.level[level + 1]].tocsr()
        for i in range(len(index) - 1):
            start, end = [index[i], index[i + 1]]
            batch_datas = FloatTensor(self.datas[start:end])
            batch_labels = FloatTensor(label_level[start:end].toarray())
            yield batch_datas, batch_labels

    def number_of_data_in_each_class(self):
        if self.state != "embedding":
            raise NotEmbeddingState
        return np.sum(self.labels, 0).astype(int).tolist()[0]

    def number_of_data(self):
        return len(self.datas)

    def index_of_level(self, level):
        return self.level[level], self.level[level + 1]

    def size_of_feature(self):
        return self.datas.shape[1]
