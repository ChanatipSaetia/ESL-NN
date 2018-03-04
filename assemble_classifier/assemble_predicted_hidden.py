from assemble_classifier import AssembleLevel
from classifier import LCPLNoLabel, LCPLPredicted_Hidden, LCPLNoLabelHidden
import torch
from torch.autograd import Variable
from torch import FloatTensor
import os
import pickle


class AssemblePredictedHidden(AssembleLevel):

    def __init__(self, data_name, dataset, dataset_validate, dataset_test, iteration, batch_size, hidden_size, learning_rate=0.001, use_dropout=True, early_stopping=True, stopping_time=500, start_level=0, end_level=10000):
        super(AssemblePredictedHidden, self).__init__(data_name, dataset, dataset_validate, dataset_test, iteration, batch_size,
                                                      hidden_size, learning_rate, use_dropout, early_stopping, stopping_time, start_level, end_level)

    def initial_classifier(self):
        torch.manual_seed(12345)
        for level in range(self.dataset.number_of_level()):
            if level == 0:
                self.initial_first_classifier(level)
            else:
                self.initial_other_classifier(level)
            # initial weight
            level = self.dataset.index_of_level(level)
            level_count = self.dataset.number_of_data_in_each_class()[
                level[0]:level[1]]
            number_of_data = self.dataset.number_of_data()
            self.classifier[-1].initial_weight(number_of_data, level_count)

    def initial_first_classifier(self, level):
        # create classifier
        input_size = self.dataset.size_of_feature()
        number_of_class = self.dataset.check_each_number_of_class(level)
        hidden = number_of_class * 2 if number_of_class * 2 <= 300 else 300
        model = LCPLNoLabelHidden(
            input_size, hidden, number_of_class, use_dropout=self.use_dropout, learning_rate=self.learning_rate)
        if torch.cuda.is_available():
            model = model.cuda()
        self.classifier.append(model)

    def initial_other_classifier(self, level):
        # create classifier
        input_size = self.dataset.size_of_feature()
        prev_number_of_class = self.classifier[level - 1].hidden_size
        number_of_class = self.dataset.check_each_number_of_class(level)
        hidden = number_of_class * 2 if number_of_class * 2 <= 300 else 300
        model = LCPLPredicted_Hidden(
            input_size, prev_number_of_class, hidden, number_of_class, use_dropout=self.use_dropout, learning_rate=self.learning_rate)
        if torch.cuda.is_available():
            model = model.cuda()
        self.classifier.append(model)

    def input_classifier(self, x, level, batch_number, mode):
        if level == 0:
            return x
        else:
            input_directory = "data/%s/output/%d/%s" % (
                self.data_name, (level - 1), mode)
            if not os.path.exists(input_directory):
                os.makedirs(input_directory)

            if not os.path.isfile(input_directory + '/%d.pickle' % batch_number):
                input_data = self.input_classifier(
                    x, level - 1, batch_number, mode)
                if torch.cuda.is_available():
                    input_data = input_data.cuda()
                input_data = Variable(input_data, volatile=True)
                prev_pred = self.classifier[level -
                                            1].forward_hidden(input_data).data.cpu().numpy()
                with open(input_directory + '/%d.pickle' % batch_number, 'wb') as f:
                    pickle.dump(prev_pred, f)

            else:
                with open(input_directory + '/%d.pickle' % batch_number, 'rb') as f:
                    prev_pred = pickle.load(f)
            prev_pred = FloatTensor(prev_pred)
            return torch.cat([x, prev_pred], 1)
