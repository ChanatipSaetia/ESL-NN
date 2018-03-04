from .Doc2Vec import GensimDoc2Vec
from gensim.models.doc2vec import TaggedDocument


class NoTag_Doc2Vec(GensimDoc2Vec):
    def prepare_data(self, datas, labels):
        return [TaggedDocument(
            datas[i], [str(i)]) for i in range(len(datas))]
