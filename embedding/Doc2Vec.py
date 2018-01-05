from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import sys


class GensimDoc2Vec():

    def __init__(self, number_of_class, alpha=0.001, size=25, dm=0, window=8, min_count=3, epoch=200):
        print("Doc2Vec by Gensim")
        self.alpha = alpha
        self.dm = dm
        self.window = window
        self.min_count = min_count
        self.size = size
        self.epoch = epoch
        self.model = Doc2Vec(alpha=self.alpha, size=self.size, dm=self.dm, window=self.window,
                             min_count=self.min_count, workers=4, compute_loss=True, seed=12345)
        self.all_label = set(range(number_of_class))

    def tag_vector(self):
        tag_vector = []
        for i in range(len(self.all_label)):
            try:
                tag_vector.append(self.model.docvecs['class_%d' % i])
            except KeyError:
                tag_vector.append(np.zeros(self.size))
        tag_vector = np.array(tag_vector)
        return tag_vector

    def calculate_similar(self, transform_data, labels, tag_vector):
        cosine = cosine_similarity(transform_data, tag_vector)

        all_s = []
        all_d = []
        for i, label in zip(range(len(transform_data)), labels):
            same = np.mean(cosine[i][list(label)])
            diff = np.mean(cosine[i][list(self.all_label - set(label))])
            all_s.append(same)
            all_d.append(diff)

        diff = (np.mean(np.array(all_s)) - np.mean(np.array(all_d))) / \
            np.mean(np.array(all_s))

        return diff

    def fit(self, datas, labels, datas_validate, labels_validate):
        documents = [TaggedDocument(
            datas[i], [str(i)] + ['class_%d' % j for j in labels[i]]) for i in range(len(datas))]
        self.model.build_vocab(documents)
        max_diff = 0
        each_time = int(self.epoch / 50)
        time_before_stop = self.epoch / (each_time * 5)
        c = 0
        is_saving = False
        for i in range(int(self.epoch / each_time)):
            self.model.train(
                documents, total_examples=self.model.corpus_count, epochs=each_time)

            tag_vector = self.tag_vector()
            transform_data = self.transform(datas_validate)
            diff = self.calculate_similar(
                transform_data, labels_validate, tag_vector)
            print("Epoch: %i Similar: %.2f" % ((i + 1) * each_time, diff))
            if max_diff < diff:
                max_diff = diff
                c = 0
                self.model.save('best_now/doc2vec.model')
                is_saving = True
            elif i >= time_before_stop:
                c = c + 1

            if c >= 2:
                print("Stopping Similar: %.2f" % diff)
                if is_saving:
                    self.model = Doc2Vec.load('best_now/doc2vec.model')
                break

        return self

    def transform(self, datas):
        return np.array([self.model.infer_vector(i) for i in datas])