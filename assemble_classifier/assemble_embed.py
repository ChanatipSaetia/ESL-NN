from assemble_classifier import AssembleLevel
from classifier import LCPLEmbed
import torch
import numpy as np
import torch
from torch import ByteTensor, FloatTensor
from torch.autograd import Variable
import sys
from torch.optim.lr_scheduler import ReduceLROnPlateau

from evaluation import f1_from_tp_pcp, tp_pcp


class AssembleEmbed(AssembleLevel):

    def __init__(self, data_name, dataset, dataset_validate, dataset_test, iteration, batch_size, embed_size, hidden_size, learning_rate=0.001, use_dropout=True, early_stopping=True, stopping_time=500, start_level=0, end_level=10000):
        self.embed_size = embed_size
        super(AssembleEmbed, self).__init__(data_name, dataset, dataset_validate, dataset_test, iteration, batch_size,
                                            hidden_size, learning_rate, use_dropout, early_stopping, stopping_time, start_level, end_level)

    def initial_classifier(self):
        torch.manual_seed(12345)
        for level in range(self.dataset.number_of_level()):
            # create classifier
            input_size = self.dataset.size_of_feature()
            number_of_class = self.dataset.check_each_number_of_class(level)
            model = LCPLEmbed(
                input_size, self.embed_size, self.hidden_size[level], number_of_class, use_dropout=self.use_dropout, learning_rate=self.learning_rate)
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

    def pretrain_loading(self):
        number_of_data = self.dataset.number_of_data()
        count = self.dataset.number_of_data_in_each_class()
        input_size = self.dataset.size_of_feature()
        number_of_class = self.dataset.number_of_classes()
        self.pretrain_model = LCPLEmbed(
            input_size, self.embed_size, 0, number_of_class, use_dropout=False, learning_rate=0.01)
        self.pretrain_model.initial_weight(number_of_data, count)
        max_f1_macro = 0
        c = 0
        iteration = 2000
        stopping = 200
        # scheduler = ReduceLROnPlateau(
        #     self.pretrain_model.optimizer, mode='max', patience=50, factor=0.1, threshold=1e-3)
        for epoch in range(iteration):
            all_loss = 0
            number_of_batch = 0
            all_batch = np.arange(
                0, self.dataset.number_of_data(), self.batch_size).shape[0]
            for datas, labels in self.dataset.generate_batch(-1, self.batch_size):
                number_of_batch = number_of_batch + 1
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                datas_in = self.input_classifier(
                    datas, -1, number_of_batch, "train")
                if torch.cuda.is_available():
                    datas_in = datas_in.cuda()
                    labels = labels.cuda()
                datas_in = Variable(datas_in)
                labels_in = Variable(labels)
                loss = self.pretrain_model.train_model(datas_in, labels_in)
                all_loss = all_loss + loss
                sys.stdout.write("\rLevel: %.3f Epoch: %d/%d Batch: %d/%d Loss: %.3f " %
                                 (0, epoch + 1, iteration, number_of_batch, all_batch,
                                     all_loss / number_of_batch))
                sys.stdout.flush()
            f1_macro, _ = self.evaluate_each_level(-1, "validate")
            con = (max_f1_macro < f1_macro)
            if(con and epoch >= int(iteration / 30)):
                max_f1_macro = f1_macro
                c = 0
                torch.save(self.pretrain_model, "best_now/%s/model_%d.model" %
                           (self.data_name, 0))
            elif(epoch >= int(iteration / 30)):
                c = c + 1

            # scheduler.step(f1_macro)
            # print(self.pretrain_model.optimizer.param_groups[0]['lr'])

            if(c >= stopping and self.early_stopping):
                train_f1_macro, _ = self.evaluate_each_level(-1, "train")
                print("Training F1 macro: %.3f Validate F1 macro: %.3f" %
                      (train_f1_macro, f1_macro))
                self.pretrain_model = torch.load("best_now/%s/model_%d.model" %
                                                 (self.data_name, 0))
                break

            if(epoch % int(iteration / 30) == int(iteration / 30) - 1):
                train_f1_macro, _ = self.evaluate_each_level(-1, "train")
                print("Training F1 macro: %.3f Validate F1 macro: %.3f" %
                      (train_f1_macro, f1_macro))
        torch.save(self.pretrain_model.embed, "best_now/%s/embed.model" %
                   (self.data_name))
        for level, model in enumerate(self.classifier):
            model.embed = torch.load("best_now/%s/embed.model" %
                                     (self.data_name))
        print()

    def evaluate_each_level(self, level, mode, threshold=0):
        if mode == "train":
            evaluated_data = self.dataset
        elif mode == "validate":
            evaluated_data = self.dataset_validate
        elif mode == "test":
            evaluated_data = self.dataset_test

        if level == -1:
            model = self.pretrain_model
        else:
            model = self.classifier[level]

        number_of_class = model.number_of_class
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
            tp, pcp, cp = model.evaluate_tp_pcp(
                datas_in, labels_in, threshold)
            all_tp = all_tp + tp
            all_pcp = all_pcp + pcp
            all_cp = all_cp + cp
        f1_macro, f1_micro = f1_from_tp_pcp(
            all_tp, all_pcp, all_cp, number_of_class)
        f1_macro = f1_macro.data.cpu().numpy()[0]
        f1_micro = f1_micro.data.cpu().numpy()[0]
        return f1_macro, f1_micro
