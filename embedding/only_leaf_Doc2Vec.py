from .Doc2Vec import GensimDoc2Vec
from gensim.models.doc2vec import TaggedDocument


class OnlyLeafDoc2Vec(GensimDoc2Vec):
    def prepare_data(self, datas, labels):
        return [TaggedDocument(
            datas[i], [str(i)]+ ['class_%d' % sorted(list(labels[i]))[-1]]) for i in range(len(datas))]
