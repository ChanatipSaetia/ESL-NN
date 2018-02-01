from classifier import EachLevelClassifier
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F


class LCPLEmbed(EachLevelClassifier):

    def __init__(self, input_size, embed_size, hidden_size, number_of_class, use_dropout=True, learning_rate=0.001):
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.tf = [0] * input_size
        super(LCPLEmbed, self).__init__(input_size,
                                        number_of_class, use_dropout, learning_rate)

    def initial_structure(self):
        self.embed = nn.Embedding(
            self.input_size, self.embed_size)
        if self.hidden_size != 0:
            self.dense = nn.Linear(self.embed_size, self.hidden_size)
        if self.use_dropout:
            self.dropout_input = nn.Dropout(p=0.15)
            self.dropout = nn.Dropout(p=0.35)
        input_logit = self.hidden_size if self.hidden_size != 0 else self.embed_size
        self.logit = nn.Linear(input_logit, self.number_of_class)

    def initial_tf(self, feature_count):
        self.tf = feature_count

    def forward(self, x):
        padding = (x != 0)
        count = torch.sum(padding, 1).view(-1, 1).float()
        padding = padding.view(-1, x.shape[1], 1).float()
        x = self.embed(x)
        x = padding * x
        x = torch.sum(x, 1) / count
        if self.use_dropout:
            x = self.dropout_input(x)
        if self.hidden_size != 0:
            x = self.dense(x)
            x = F.relu(x)
            if self.use_dropout:
                x = self.dropout(x)
        x = self.logit(x)
        return x
