import os
import shutil
import sys

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from torch import ByteTensor, FloatTensor
from torch.autograd import Variable
from scipy.sparse import csr_matrix

from evaluation import f1_from_tp_pcp, tp_pcp


class AssembleLevel():

    def __init__(self, data_name, dataset, dataset_validate, dataset_test, iteration, batch_size, hidden_size,
                 learning_rate=0.001, use_dropout=True, early_stopping=True, stopping_time=500, start_level=0, end_level=10000):
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
        self.start_level = start_level
        self.end_level = end_level
        self.learning_rate = learning_rate
        self.classifier = []
        self.initial_classifier()
        self.initial_weight()

    def initial_classifier(self):
        raise NotImplementedError

    def input_classifier(self, x, level, batch_number, mode):
        raise NotImplementedError

    def initial_weight(self):
        for level, model in enumerate(self.classifier):
            number_of_data = self.dataset.number_of_data()
            index = self.dataset.index_of_level(level)
            count = self.dataset.number_of_data_in_each_class()[
                index[0]:index[1]]
            model.initial_weight(number_of_data, count)

    def pretrain_loading(self):
        pass

    def train(self):
        self.pretrain_loading()
        for level, model in enumerate(self.classifier):
            max_f1_macro = 0
            c = 0
            if level < self.start_level:
                self.classifier[level] = torch.load("best_now/%s/model_%d.model" %
                                                    (self.data_name, level))
                continue
            else:
                if os.path.exists('data/%s/output/%d' % (self.data_name, level)):
                    shutil.rmtree('data/%s/output/%d' %
                                  (self.data_name, level))
            if level >= self.end_level:
                break
            torch.save(model, "best_now/%s/model_%d.model" %
                       (self.data_name, level))

            # scheduler = ReduceLROnPlateau(
            #     self.pretrain_model.optimizer, mode='max', patience=30, factor=0.1, threshold=1e-3)
            for epoch in range(self.iteration):
                all_loss = 0
                number_of_batch = 0
                all_batch = np.arange(
                    0, self.dataset.number_of_data(), self.batch_size).shape[0]
                for datas, labels in self.dataset.generate_batch(level, self.batch_size):
                    number_of_batch = number_of_batch + 1
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    datas_in = self.input_classifier(
                        datas, level, number_of_batch, "train")
                    if torch.cuda.is_available():
                        datas_in = datas_in.cuda()
                        labels = labels.cuda()
                    datas_in = Variable(datas_in)
                    labels_in = Variable(labels)
                    loss = model.train_model(datas_in, labels_in)
                    all_loss = all_loss + loss
                    sys.stdout.write("\rLevel: %.3f Epoch: %d/%d Batch: %d/%d Loss: %.3f " %
                                     (level + 1, epoch + 1, self.iteration, number_of_batch, all_batch,
                                      all_loss / number_of_batch))
                    sys.stdout.flush()
                f1_macro, _ = self.evaluate_each_level(level, "validate")
                con = (max_f1_macro < f1_macro)
                each_print = int(self.iteration / 30)
                each_print = 1 if each_print == 0 else each_print
                if(con):
                    max_f1_macro = f1_macro
                    c = 0
                    torch.save(model, "best_now/%s/model_%d.model" %
                               (self.data_name, level))
                elif(epoch >= each_print):
                    c = c + 1

                # scheduler.step(f1_macro)

                if(c >= self.stopping_time and self.early_stopping):
                    train_f1_macro, _ = self.evaluate_each_level(
                        level, "train")
                    print("Stopping F1 macro: %.3f Validate F1 macro: %.3f" %
                          (train_f1_macro, max_f1_macro))
                    self.classifier[level] = torch.load("best_now/%s/model_%d.model" %
                                                        (self.data_name, level))
                    break

                if(epoch % each_print == each_print - 1):
                    train_f1_macro, _ = self.evaluate_each_level(
                        level, "train")
                    print("Training F1 macro: %.3f Validate F1 macro: %.3f" %
                          (train_f1_macro, f1_macro))
            print()

    def evaluate_each_level(self, level, mode, threshold=0):
        if mode == "train":
            evaluated_data = self.dataset
        elif mode == "validate":
            evaluated_data = self.dataset_validate
        elif mode == "test":
            evaluated_data = self.dataset_test

        number_of_class = self.classifier[level].number_of_class
        initial_tp = FloatTensor([0] * number_of_class)
        initial_pcp = FloatTensor([0] * number_of_class)
        initial_cp = FloatTensor([0] * number_of_class)
        if torch.cuda.is_available():
            initial_tp = initial_tp.cuda()
            initial_pcp = initial_pcp.cuda()
            initial_cp = initial_cp.cuda()
        all_tp = Variable(initial_tp)
        all_pcp = Variable(initial_pcp)
        all_cp = Variable(initial_cp)

        number_of_batch = 0
        for datas, labels in evaluated_data.generate_batch(level, self.batch_size):
            number_of_batch = number_of_batch + 1
            datas_in = self.input_classifier(
                datas, level, number_of_batch, mode)
            if torch.cuda.is_available():
                datas_in = datas_in.cuda()
                labels = labels.cuda()
            datas_in = Variable(datas_in, volatile=True)
            labels_in = Variable(labels, volatile=True)
            tp, pcp, cp = self.classifier[level].evaluate_tp_pcp(
                datas_in, labels_in, threshold)
            all_tp = all_tp + tp
            all_pcp = all_pcp + pcp
            all_cp = all_cp + cp
        f1_macro, f1_micro = f1_from_tp_pcp(
            all_tp, all_pcp, all_cp, number_of_class)
        f1_macro = f1_macro.data.cpu().numpy()[0]
        f1_micro = f1_micro.data.cpu().numpy()[0]
        return f1_macro, f1_micro

    def evaluate(self, mode, correction=True, mandory_leaf=False):
        if mode == "train":
            evaluated_data = self.dataset
        elif mode == "validate":
            evaluated_data = self.dataset_validate
        elif mode == "test":
            evaluated_data = self.dataset_test

        number_of_batch = 0
        number_of_class = self.dataset.number_of_classes()
        all_tp = FloatTensor([0] * number_of_class)
        all_pcp = FloatTensor([0] * number_of_class)
        all_cp = FloatTensor([0] * number_of_class)
        if torch.cuda.is_available():
            all_tp = all_tp.cuda()
            all_pcp = all_pcp.cuda()
            all_cp = all_cp.cuda()

        for datas, labels in evaluated_data.generate_batch(-1, self.batch_size):

            number_of_batch = number_of_batch + 1
            all_labels = FloatTensor([])
            if mandory_leaf:
                all_pred = FloatTensor([])
            else:
                all_pred = ByteTensor([])
            if torch.cuda.is_available():
                all_labels = all_labels.cuda()
                all_pred = all_pred.cuda()
            for level in range(self.dataset.number_of_level()):
                datas_in = self.input_classifier(
                    datas, level, number_of_batch, mode)

                datas_in = Variable(datas_in, volatile=True)
                each_level = labels[:, self.dataset.level[level]:self.dataset.level[level + 1]]
                if torch.cuda.is_available():
                    datas_in = datas_in.cuda()
                    each_level = each_level.cuda()
                if mandory_leaf:
                    pred = self.classifier[level](
                        datas_in).data
                else:
                    pred = self.classifier[level].output_with_threshold(
                        datas_in).data
                all_labels = torch.cat((all_labels, each_level), 1)
                all_pred = torch.cat((all_pred, pred), 1)

            if mandory_leaf:
                all_pred = self.get_leaf_node(all_pred)
                all_pred = self.to_one_hot(all_pred)
            if correction:
                all_pred = self.child_based_correction(all_pred)
            tp, pcp, cp = tp_pcp(all_labels, all_pred, use_threshold=False)
            # print(tp, all_tp)
            all_tp = all_tp + tp
            all_pcp = all_pcp + pcp
            all_cp = all_cp + cp

        f1_macro, f1_micro = f1_from_tp_pcp(
            all_tp, all_pcp, all_cp, self.dataset.number_of_classes())
        f1_each_level = []
        for level in range(self.dataset.number_of_level()):
            each_tp = all_tp[self.dataset.level[level]:self.dataset.level[level + 1]]
            each_pcp = all_pcp[self.dataset.level[level]:self.dataset.level[level + 1]]
            each_cp = all_cp[self.dataset.level[level]:self.dataset.level[level + 1]]
            each_f1_macro, each_f1_micro = f1_from_tp_pcp(
                each_tp, each_pcp, each_cp, self.classifier[level].number_of_class)
            f1_each_level.append((each_f1_macro, each_f1_micro))
        return f1_macro, f1_micro, f1_each_level

    def get_leaf_node(self, y):
        num_test = F.sigmoid(y).cpu().numpy()

        for l, m in enumerate(self.classifier):
            first = self.dataset.level[l]
            last = self.dataset.level[l + 1]
            num_test[:, first:last] = num_test[:, first:last]

        for h in self.dataset.hierarchy:
            child = list(self.dataset.hierarchy[h])
            num_test[:, child] = num_test[:, child] * \
                np.repeat(num_test[:, h:h + 1], len(child), axis=1)

        only_leaf = num_test * \
            np.invert(self.dataset.not_leaf_node).astype(float)
        sort_only_leaf = only_leaf.argsort()
        indice_sort = sort_only_leaf[:, -self.dataset.max_label:]
        distribution = np.sum(
            np.array(list(map((lambda x, y: x[y]), only_leaf, indice_sort))), 1)
        mean_dis = np.mean(distribution)
        sd_dis = np.std(distribution)

        leaf_in_each_row = np.apply_along_axis(
            lambda d: ((d - mean_dis) / sd_dis) *
            self.dataset.sd_label + self.dataset.mean_label,
            0, distribution).astype(int)
        ans_index = list(
            map((lambda x, y: x[-y:]), indice_sort, leaf_in_each_row))
        return ans_index

    def to_one_hot(self, ans_index):
        indice = [j for i in ans_index for j in i]
        indptr = np.cumsum([0] + [len(i) for i in ans_index])
        data_one = np.ones(len(indice))
        return csr_matrix((data_one, indice, indptr), shape=(
            len(ans_index), self.dataset.number_of_classes())).toarray()

    def export_result(self, mode, correction=True, mandory_leaf=False):
        if mode == "train":
            evaluated_data = self.dataset
        elif mode == "validate":
            evaluated_data = self.dataset_validate
        elif mode == "test":
            evaluated_data = self.dataset_test

        number_of_batch = 0
        for datas, labels in evaluated_data.generate_batch(-1, self.batch_size):

            number_of_batch = number_of_batch + 1
            all_labels = FloatTensor([])
            if mandory_leaf:
                all_pred = FloatTensor([])
            else:
                all_pred = ByteTensor([])
            if torch.cuda.is_available():
                all_labels = all_labels.cuda()
                all_pred = all_pred.cuda()
            for level in range(self.dataset.number_of_level()):
                datas_in = self.input_classifier(
                    datas, level, number_of_batch, mode)

                datas_in = Variable(datas_in, volatile=True)
                each_level = labels[:, self.dataset.level[level]
                    :self.dataset.level[level + 1]]
                if torch.cuda.is_available():
                    datas_in = datas_in.cuda()
                    each_level = each_level.cuda()
                if mandory_leaf:
                    pred = self.classifier[level](
                        datas_in).data
                else:
                    pred = self.classifier[level].output_with_threshold(
                        datas_in).data
                all_labels = torch.cat((all_labels, each_level), 1)
                all_pred = torch.cat((all_pred, pred), 1)

            if mandory_leaf:
                all_pred = self.get_leaf_node(all_pred)
            else:
                all_pred = self.child_based_correction(all_pred)
                all_pred = self.select_deepest_label(all_pred)
                # all_pred = self.to_one_hot(all_pred)
            # if correction:

            if not os.path.exists("export/%s/%s" % (self.data_name, mode)):
                os.makedirs("export/%s/%s" % (self.data_name, mode))
            np_all_name = np.array(self.dataset.all_name)
            with open("export/%s/%s/result.txt" % (self.data_name, mode), 'a') as f:
                for p in all_pred:
                    f.write(" ".join(np_all_name[p]) + "\n")
                f.close()

    def child_based_correction(self, y):
        if type(y) != np.ndarray:
            num_test = y.cpu().numpy()
        else:
            num_test = y
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

    def select_deepest_label(self, y):
        num_test = y.cpu().numpy()
        for k in num_test:
            for n in range(self.dataset.number_of_level()):
                start_index = self.dataset.level[n]
                last_index = self.dataset.level[n + 1]
                for i in range(start_index, last_index):
                    if i in self.dataset.hierarchy:
                        for c in self.dataset.hierarchy[i]:
                            if k[c]:
                                k[i] = 0
        num_test = num_test.astype(bool)
        return num_test

    def tuning_threshold(self):
        for level, model in enumerate(self.classifier):
            if level >= self.end_level:
                break
            best_threshold = 0.5
            max_f1_macro = 0
            for t in np.arange(0.05, 0.90, 0.05):
                f1_macro, _ = self.evaluate_each_level(level, "validate", t)
                if max_f1_macro < f1_macro:
                    best_threshold = t
                    max_f1_macro = f1_macro
            model.best_threshold = best_threshold
            model.change_ratio = 0.5 / best_threshold
