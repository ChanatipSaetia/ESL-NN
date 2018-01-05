import torch
import torch.nn.functional as F
from torch import FloatTensor


def f1_per_class(labels, pred, index, level):
    for i, label_level in enumerate(level):
        number_of_class = len(label_level)
        each_label = FloatTensor(labels[:, label_level]).cuda()
        first = 0
        if i != 0:
            first = index[i - 1]
        corre = pred[:, first:index[i]]
        print(f1_score(each_label, corre, number_of_class, use_threshold=False))


def give_tp_pcp(label, pred):
    true_pos = torch.sum((label * (label == pred)).float(), 0)
    pred_con_pos = torch.sum(pred.float(), 0)
    con_pos = torch.sum(label.float(), 0)
    return true_pos.float(), pred_con_pos.float(), con_pos.float()


def f1_score(label, pred, number_of_class, use_threshold=True, threshold=0.5):
    label = label.byte()
    if use_threshold:
        pred = (F.sigmoid(pred) >= threshold)
    else:
        pred = pred.byte()
    true_pos, pred_con_pos, con_pos = give_tp_pcp(label, pred)
    # f1_micro
    try:
        precision = torch.sum(true_pos) / torch.sum(pred_con_pos)
        recall = torch.sum(true_pos) / torch.sum(con_pos)
        f1_micro = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        f1_micro = 0

    # f1_macro
    each_precision = true_pos / pred_con_pos
    each_recall = true_pos / con_pos
    before_sum = 2 * each_precision * \
        each_recall / (each_precision + each_recall)
    before_sum[before_sum != before_sum] = 0
    f1_macro_score = sum(before_sum) / number_of_class

    return f1_macro_score, f1_micro


def f1_macro(label, data, number_of_class, threshold=0.5):
    label = label.byte()
    pred = (F.sigmoid(data) >= threshold)
    true_pos, pred_con_pos, con_pos = give_tp_pcp(label, pred)

    # f1_macro
    each_precision = true_pos / pred_con_pos
    each_recall = true_pos / con_pos
    before_sum = 2 * each_precision * \
        each_recall / (each_precision + each_recall)
    before_sum[before_sum != before_sum] = 0
    f1_macro_score = sum(before_sum) / number_of_class

    return f1_macro_score
