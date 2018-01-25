import os
import math
import pickle
from functools import reduce
from itertools import repeat

import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

import data.exception as ex
import data.hierarchy as hie


def import_each_row(row):
    label, data = row.strip().split(":")
    if data[1:-1] == "":
        raise ex.NoFeatureInRow
    data = data[1:-1].split(",")
    if label == '':
        raise ex.NoLabelInRow
    label = sorted([i for i in label.split(',')])
    return data, label


def map_data_to_list(string):
    split = string.split(":")
    if len(split) == 1:
        return []
    return [split[0]] * int(split[1])


def import_each_row_bag_of_word(row):
    split_comma = row.strip().split(",")
    labels = list(map(lambda x: x.strip(), split_comma[:-1]))
    split_space = split_comma[-1].strip().split(" ")
    if len(split_space) == 1 and len(split_space[0].strip().split(":")) == 1:
        raise ex.NoFeatureInRow
    if len(split_comma) == 1 and len(split_space[0].strip().split(":")) != 1:
        raise ex.NoLabelInRow

    labels.append(split_space[0].strip())
    data = list(reduce((lambda x, y: x + y),
                       map(map_data_to_list, split_space[1:])))
    labels.sort()
    return data, labels


def import_data_sequence(file_name):
    datas = []
    labels = []
    with open('data/%s' % file_name) as files:
        for row in files:
            try:
                data, label = import_each_row(row)
                datas.append(data)
                labels.append(label)
            except ex.NoFeatureInRow:
                pass
            except ex.NoLabelInRow:
                pass
    return datas, labels


def import_data_bag_of_word(file_name):
    datas = []
    labels = []
    with open('data/%s' % file_name) as files:
        for row in files:
            try:
                data, label = import_each_row_bag_of_word(row)
                datas.append(data)
                labels.append(label)
            except ex.NoFeatureInRow:
                pass
            except ex.NoLabelInRow:
                pass
    return datas, labels


def import_data(file_name, sequence=True):
    if sequence:
        return import_data_sequence(file_name)
    else:
        return import_data_bag_of_word(file_name)


def each_label(parent_of, i):
    try:
        all_p = parent_of[i]
        all_label = set([i])
        for p in all_p:
            all_label = all_label | each_label(parent_of, p)
        return all_label
    except KeyError:
        return set([i])


def each_row_of_label(parent_of, name_to_index, label):
    new_index = list(map(lambda x: name_to_index[x], label))
    all_label = list(map(each_label, repeat(parent_of), new_index))
    return reduce((lambda x, y: x | y), all_label)


def map_index_of_label(file_name, labels):
    _, parent_of, _, name_to_index, _ = hie.load_hierarchy(file_name)
    return list(map(each_row_of_label, repeat(parent_of), repeat(name_to_index), labels))


def load_data_in_pickle(file_name):
    with open('data/%s' % file_name, 'rb') as f:
        data, label = pickle.load(f)
    return data, label


def save_data_in_pickle(file_name, datas, labels):
    directory = "data/%s" % "/".join(file_name.split("/")[:-1])
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open('data/%s' % file_name, 'wb') as f:
        pickle.dump([datas, labels], f)


def split_data(datas, labels, data_name):
    directory = "data/%s/fold" % data_name
    if not os.path.exists(directory):
        os.makedirs(directory)
    kf = KFold(n_splits=5, random_state=12345)
    i = 1
    datas = np.array(datas)
    labels = np.array(labels)
    for train, test in kf.split(datas):
        train_data, validate_data, train_target, validate_target = train_test_split(
            datas[train], labels[train], test_size=0.1, random_state=12345)
        with open('data/%s/fold/data_%d.pickle.train' % (data_name, i), 'wb') as f:
            pickle.dump([train_data, train_target], f)
            f.close()
        with open('data/%s/fold/data_%d.pickle.validate' % (data_name, i), 'wb') as f:
            pickle.dump([validate_data, validate_target], f)
            f.close()
        with open('data/%s/fold/data_%d.pickle.test' % (data_name, i), 'wb') as f:
            pickle.dump([datas[test], labels[test]], f)
            f.close()
        i = i + 1


def split_data_lshtc(data_name, least_data=5):
    directory = "data/%s/pickle" % data_name
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_name = "%s/data.txt" % (data_name)
    file_name_test = "%s/data_test.txt" % (data_name)
    datas, labels = import_data(file_name, sequence=False)
    datas_test, labels_test = import_data(file_name_test, sequence=False)
    hierarchy_file_name = "%s/hierarchy.pickle" % data_name
    new_labels = map_index_of_label(
        hierarchy_file_name, labels)
    train_data, validate_data, train_target, validate_target = split_validate(
        datas, new_labels, hierarchy_file_name, least_data)
    with open('data/%s/pickle/data.pickle.train' % (data_name), 'wb') as f:
        pickle.dump([train_data, train_target], f)
        f.close()
    with open('data/%s/pickle/data.pickle.validate' % (data_name), 'wb') as f:
        pickle.dump([validate_data, validate_target], f)
        f.close()
    with open('data/%s/pickle/data.pickle.test' % (data_name), 'wb') as f:
        pickle.dump([np.array(datas_test), labels_test], f)
        f.close()


def split_validate(datas, labels, hierarchy_name, least_data=5):
    train_index, validate_index = create_train_validate_index(
        labels, hierarchy_name, least_data)
    datas = np.array(datas)
    labels = np.array(labels)
    train_data, train_target = datas[list(
        train_index)], labels[list(train_index)]
    validate_data, validate_target = datas[list(
        validate_index)], labels[list(validate_index)]
    return train_data, train_target, validate_data, validate_target


def create_train_validate_index(labels, hierachy_name, least_data=5):
    _, _, all_name, _, _ = hie.load_hierarchy(hierachy_name)
    map_data_index = create_map_data_index(labels)
    train = set()
    validate = set()
    for i in range(len(all_name)):
        try:
            each_label_map = map_data_index[i]
        except KeyError:
            continue
        if len(each_label_map) < least_data:
            continue
        middle_index = math.floor((len(each_label_map) * 4) / 5)
        train.update(each_label_map[:middle_index])
        validate.update(each_label_map[middle_index:])
    return train, validate


def create_map_data_index(labels):
    map_data_index = {}
    for i, label in enumerate(labels):
        for l in label:
            try:
                map_data_index[l].append(i)
            except KeyError:
                map_data_index[l] = [i]
    return map_data_index
