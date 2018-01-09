import os
import sys

import numpy as np
import torch
from torch import FloatTensor, ByteTensor
from torch.autograd import Variable

from evaluation import f1_from_tp_pcp, tp_pcp


class AssembleLevel():

    def __init__(self, data_name, dataset, dataset_validate, dataset_test, iteration, batch_size, hidden_size, use_dropout=True, early_stopping=True, stopping_time=500):
        self.data_name = data_name
        if not os.path.exists("best_now/%s" % data_name):
            os.makedirs("best_now/%s" % data_name)
        self.dataset = dataset
        self.dataset_validate = dataset_validate
        self.dataset_test = dataset_test
        self.iteration = iteration
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.use_dropout = use_dropout
        self.early_stopping = early_stopping
        self.stopping_time = stopping_time
        self.classifier = []
        self.initial_classifier()
        self.initial_weight()

    def initial_classifier(self):
        raise NotImplementedError

    def input_classifier(self, x, level):
        raise NotImplementedError

    def initial_weight(self):
        for level, model in enumerate(self.classifier):
            number_of_data = self.dataset.number_of_data()
            index = self.dataset.index_of_level(level)
            count = self.dataset.number_of_data_in_each_class()[
                index[0]:index[1]]
            model.initial_weight(number_of_data, count)

    def train(self):
        for level, model in enumerate(self.classifier):
            max_f1_macro = 0
            c = 0
            torch.save(model, "best_now/%s/model_%d.model" %
                       (self.data_name, level))
            for epoch in range(self.iteration):
                all_loss = 0
                number_of_batch = 0
                all_batch = np.arange(
                    0, self.dataset.number_of_data(), self.batch_size).shape[0]
                for datas, labels in self.dataset.generate_batch(level, self.batch_size):
                    # torch.empty_cache()
                    datas_in = Variable(self.input_classifier(datas, level))
                    labels_in = Variable(labels)
                    loss = model.train_model(datas_in, labels_in)
                    all_loss = all_loss + loss
                    number_of_batch = number_of_batch + 1
                    sys.stdout.write("\rLevel: %.3f Epoch: %d/%d Batch: %d/%d Loss: %.3f " %
                                     (level + 1, epoch + 1, self.iteration, number_of_batch, all_batch,
                                      all_loss / number_of_batch))
                    sys.stdout.flush()
                f1_macro, _ = self.evaluate_each_level(level, "validate")
                con = (max_f1_macro < f1_macro)
                if(con):
                    max_f1_macro = f1_macro
                    c = 0
                    torch.save(model, "best_now/%s/model_%d.model" %
                               (self.data_name, level))
                elif(epoch >= int(self.iteration / 30)):
                    c = c + 1

                if(c >= self.stopping_time and self.early_stopping):
                    print("Training Loss: %.3f Stopping F1 macro: %.3f" %
                          (all_loss / number_of_batch, f1_macro))
                    self.classifier[level] = torch.load("best_now/%s/model_%d.model" %
                                                        (self.data_name, level))
                    break

                if(epoch % int(self.iteration / 30) == int(self.iteration / 30) - 1):
                    print("Training Loss: %.3f Validate F1 macro: %.3f" %
                          (loss, max_f1_macro))
            print()

    def evaluate_each_level(self, level, mode, threshold=0):
        if mode == "train":
            evaluated_data = self.dataset
        elif mode == "validate":
            evaluated_data = self.dataset_validate
        elif mode == "test":
            evaluated_data = self.dataset_test

        number_of_class = self.classifier[level].number_of_class
        all_tp = Variable(FloatTensor([0] * number_of_class))
        all_pcp = Variable(FloatTensor([0] * number_of_class))
        all_cp = Variable(FloatTensor([0] * number_of_class))
        for datas, labels in evaluated_data.generate_batch(level, self.batch_size):
            datas_in = Variable(self.input_classifier(
                datas, level), volatile=True)
            labels_in = Variable(labels, volatile=True)
            tp, pcp, cp = self.classifier[level].evaluate_tp_pcp(
                datas_in, labels_in, threshold)
            all_tp = all_tp + tp
            all_pcp = all_pcp + pcp
            all_cp = all_cp + cp
        f1_macro, f1_micro = f1_from_tp_pcp(
            all_tp, all_pcp, all_cp, number_of_class)
        f1_macro = f1_macro.data.numpy()[0]
        f1_micro = f1_micro.data.numpy()[0]
        return f1_macro, f1_micro

    def evaluate(self, mode):
        if mode == "train":
            evaluated_data = self.dataset
        elif mode == "validate":
            evaluated_data = self.dataset_validate
        elif mode == "test":
            evaluated_data = self.dataset_test

        for datas, labels in evaluated_data.generate_batch(-1, self.batch_size):
            all_labels = FloatTensor([])
            all_pred = ByteTensor([])
            for level in range(self.dataset.number_of_level()):
                datas_in = Variable(self.input_classifier(
                    datas, level), volatile=True)
                each_level = labels[:, self.dataset.level[level]:self.dataset.level[level + 1]]
                pred = self.classifier[level].output_with_threshold(
                    datas_in).data
                all_labels = torch.cat((all_labels, each_level), 1)
                all_pred = torch.cat((all_pred, pred), 1)

            all_pred = self.child_based_correction(all_pred)
            tp, pcp, cp = tp_pcp(all_labels, all_pred, use_threshold=False)
            f1_macro, f1_micro = f1_from_tp_pcp(
                tp, pcp, cp, self.dataset.number_of_classes())
            f1_each_level = []
            for level in range(self.dataset.number_of_level()):
                each_tp = tp[self.dataset.level[level]:self.dataset.level[level + 1]]
                each_pcp = pcp[self.dataset.level[level]:self.dataset.level[level + 1]]
                each_cp = cp[self.dataset.level[level]:self.dataset.level[level + 1]]
                each_f1_macro, each_f1_micro = f1_from_tp_pcp(
                    each_tp, each_pcp, each_cp, self.classifier[level].number_of_class)
                f1_each_level.append((each_f1_macro, each_f1_micro))
            return f1_macro, f1_micro, f1_each_level

    def child_based_correction(self, y):
        num_test = y.cpu().numpy()
        for k in num_test:
            for n in range(-1 * (self.dataset.number_of_level() - 1), 0):
                n = n * -1
                start_index = self.dataset.level[n]
                last_index = self.dataset.level[n + 1]
                for i in range(start_index, last_index):
                    for p in self.dataset.parent_of[i]:
                        if k[i]:
                            k[p] = 1
        num_test = num_test.astype(float)
        correct_test = FloatTensor(num_test)
        if torch.cuda.is_available():
            correct_test = correct_test.cuda()
        return correct_test

    def tuning_threshold(self):
        for level, model in enumerate(self.classifier):
            best_threshold = 0.5
            max_f1_macro = 0
            for t in np.arange(0.05, 0.90, 0.05):
                f1_macro, _ = self.evaluate_each_level(level, "validate", t)
                if max_f1_macro < f1_macro:
                    best_threshold = t
                    max_f1_macro = f1_macro
            model.best_threshold = best_threshold
