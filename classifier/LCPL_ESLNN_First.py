from classifier import EachLevelClassifier
from torch import nn
from torch import optim
import torch.nn.functional as F


class LCPL_ESLNN_First(EachLevelClassifier):

    def __init__(self, input_size, hidden_size, number_of_class, use_dropout=True, learning_rate=0.001):
        self.hidden_size = hidden_size
        super(LCPL_ESLNN_First, self).__init__(input_size,
                                               number_of_class, use_dropout, learning_rate)

    def initial_structure(self):
        self.dense = nn.Linear(self.input_size, self.hidden_size)
        if self.use_dropout:
            self.dropout_input = nn.Dropout(p=0.10)
            self.dropout = nn.Dropout(p=0.20)
        self.logit = nn.Linear(self.hidden_size, self.number_of_class)

    def forward(self, x):
        x = self.dense(x)
        if self.use_dropout:
            x = self.dropout_input(x)
        x = F.relu(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.logit(x)
        return x
