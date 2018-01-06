import torch
from torch import FloatTensor
import torch.nn as nn
import torch.optim as optim
from evaluation import f1_score
torch.manual_seed(12345)
# torch.cuda.manual_seed(12345)


class EachLevelClassifier(nn.Module):

    def __init__(self, input_size, number_of_class, level, use_dropout=True, learning_rate=0.001):
        super(EachLevelClassifier, self).__init__()
        self.input_size = input_size
        self.number_of_class = number_of_class
        self.level = level
        self.use_dropout = use_dropout
        self.learning_rate = learning_rate
        self.best_threshold = 0.5
        self.initial_structure()
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def initial_structure(self):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

    def train_model(self, x, y):
        self.train()
        self.optimizer.zero_grad()
        output = self(x)
        loss = self.loss_function(output, y)
        loss.backward()
        self.optimizer.step()
        output_loss = loss.data.numpy()[0]
        return output_loss

    def initial_weight(self, number_of_data, count):
        weight = []
        for i, c in enumerate(count):
            try:
                w = number_of_data / c
                weight.append(w)
            except ZeroDivisionError:
                weight.append(10000)
        self.pos_weight = FloatTensor(weight)
        self.loss_function = nn.MultiLabelSoftMarginLoss(
            size_average=True, weight=self.pos_weight)

    def evaluate(self, x, y):
        self.eval()
        output = self(x)
        f1_macro, f1_micro = f1_score(y, output, 2, use_threshold=True,
                                      threshold=self.best_threshold)
        f1_macro = f1_macro.data.numpy()[0]
        f1_micro = f1_micro.data.numpy()[0]
        return f1_macro, f1_micro
