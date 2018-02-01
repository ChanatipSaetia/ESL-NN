import torch
import torch.nn.functional as F
from torch import nn, optim

from classifier import EachLevelClassifier


class LCPLPredicted(EachLevelClassifier):

    def __init__(self, input_size, previous_number_of_class, hidden_size, target_hidden_node, number_of_class, use_dropout=True):
        self.hidden_size = hidden_size
        self.previous_number_of_class = previous_number_of_class
        self.target_hidden_node = target_hidden_node
        super(LCPLPredicted, self).__init__(input_size,
                                            number_of_class, use_dropout)

    def initial_structure(self):
        self.prev_dense = nn.Linear(
            self.previous_number_of_class, self.target_hidden_node)
        self.dense = nn.Linear(self.input_size +
                               self.target_hidden_node, self.hidden_size)
        if self.use_dropout:
            self.dropout_input = nn.Dropout(p=0.15)
            self.dropout_prev = nn.Dropout(p=0.35)
            self.dropout = nn.Dropout(p=0.5)
        self.logit = nn.Linear(self.hidden_size, self.number_of_class)

    def forward(self, x):
        start_target = x.size()[1] - self.previous_number_of_class
        prev = x[:, start_target:]
        real_x = x[:, :start_target]
        prev = F.tanh(self.prev_dense(prev))
        if self.use_dropout:
            prev = self.dropout_prev(prev)
            real_x = self.dropout_input(real_x)
        x = torch.cat([prev, real_x], 1)
        x = F.relu(self.dense(x))
        if self.use_dropout:
            x = self.dropout(x)
        x = self.logit(x)
        return x
