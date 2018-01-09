from assemble_classifier import AssembleLevel
from classifier import LCPLNoLabel


class AssembleNoLabel(AssembleLevel):

    def __init__(self, data_name, dataset, dataset_validate, dataset_test, iteration, batch_size, hidden_size, use_dropout=True, early_stopping=True, stopping_time=500):
        super(AssembleNoLabel, self).__init__(data_name, dataset, dataset_validate, dataset_test, iteration, batch_size,
                                              hidden_size, use_dropout, early_stopping, stopping_time)

    def initial_classifier(self):
        for level in range(self.dataset.number_of_level()):
            # create classifier
            input_size = self.dataset.size_of_feature()
            number_of_class = self.dataset.check_each_number_of_class(level)
            level = self.dataset.index_of_level(level)
            self.classifier.append(
                LCPLNoLabel(
                    input_size, self.hidden_size[0], number_of_class, level)
            )

            # initial weight
            level_count = self.dataset.number_of_data_in_each_class()[
                level[0]:level[1]]
            number_of_data = self.dataset.number_of_data()
            self.classifier[-1].initial_weight(number_of_data, level_count)

    def input_classifier(self, x, level):
        return x
