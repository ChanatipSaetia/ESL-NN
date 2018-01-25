import os
import pickle

import data.preparation as prep

from .Dataset import Dataset


class LSHTCDataset(Dataset):

    def __init__(self, data_name, mode="train", state="first", least_data=5):
        self.least_data = least_data
        super(LSHTCDataset, self).__init__(
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
