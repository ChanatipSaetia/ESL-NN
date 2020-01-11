from assemble_classifier import AssembleLevel
from classifier import LCPL_ESLNN_First
import torch


class LCPL(AssembleLevel):

    def __init__(self, data_name, dataset, dataset_validate, dataset_test, iteration, hidden_size, learning_rate=0.001, use_dropout=True, early_stopping=True, batch_size=None, stopping_time=500, start_level=0, end_level=10000):
        super(LCPL, self).__init__(data_name, dataset, dataset_validate, dataset_test, iteration,
                                   hidden_size, learning_rate, use_dropout, early_stopping, batch_size, stopping_time, start_level, end_level)

    def initial_classifier(self):
        torch.manual_seed(12345)
        for level in range(self.dataset.number_of_level()):
            # create classifier
            input_size = self.dataset.size_of_feature()
            number_of_class = self.dataset.check_each_number_of_class(level)
            model = LCPL_ESLNN_First(
                input_size, self.hidden_size[level], number_of_class, use_dropout=self.use_dropout)
            if torch.cuda.is_available():
                model = model.cuda()
            self.classifier.append(model)

            # initial weight
            level = self.dataset.index_of_level(level)
            level_count = self.dataset.number_of_data_in_each_class()[
                level[0]:level[1]]
            number_of_data = self.dataset.number_of_data()
            self.classifier[-1].initial_weight(number_of_data, level_count)

    def input_classifier(self, x, level, batch_number, mode):
        return x
