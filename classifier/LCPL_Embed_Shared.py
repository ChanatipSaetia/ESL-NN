from classifier import EachLevelClassifier
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F


class LCPLEmbedShared(EachLevelClassifier):

    def __init__(self, input_size, embed_size, previous_number_of_class, hidden_size, target_hidden_node, number_of_class, use_dropout=True, learning_rate=0.001):
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.previous_number_of_class = previous_number_of_class
        self.target_hidden_node = target_hidden_node
        self.tf = [0] * input_size
        super(LCPLEmbedShared, self).__init__(input_size,
                                              number_of_class, use_dropout, learning_rate)

    def initial_structure(self):
        self.embed = nn.Embedding(
            self.input_size, self.embed_size)

        self.prev_dense = nn.Linear(
            self.previous_number_of_class, self.target_hidden_node)
        self.dense = nn.Linear(self.embed_size +
                               self.target_hidden_node, self.hidden_size)
        if self.use_dropout:
            self.dropout_input = nn.Dropout(p=0.15)
            self.dropout_prev = nn.Dropout(p=0.35)
            self.dropout = nn.Dropout(p=0.5)

        self.logit = nn.Linear(self.hidden_size, self.number_of_class)

    def initial_tf(self, feature_count):
        self.tf = feature_count

    def forward(self, x):

        start_target = x.size()[1] - self.previous_number_of_class
        prev = x[:, start_target:]
        real_x = x[:, :start_target].long()
        prev = F.tanh(self.prev_dense(prev))
        if self.use_dropout:
            prev = self.dropout_prev(prev)

        padding = (real_x != 0)
        count = torch.sum(padding, 1).view(-1, 1).float()
        padding = padding.view(-1, real_x.shape[1], 1).float()
        real_x = self.embed(real_x)
        real_x = padding * real_x
        real_x = torch.sum(real_x, 1) / count
        if self.use_dropout:
            real_x = self.dropout_input(real_x)

        x = torch.cat([prev, real_x], 1)
        x = self.dense(x)
        x = F.relu(x)

        if self.use_dropout:
            x = self.dropout(x)
        x = self.logit(x)
        return x
