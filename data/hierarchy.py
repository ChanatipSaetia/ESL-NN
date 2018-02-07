import os
import pickle
import numpy as np


def create_hierarchy_structure(file_name):
    hierarchy = dict()
    parent_of = dict()
    all_name = []
    name_to_index = {}
    temp_name = set()
    with open("data/%s" % file_name) as f:
        for row in f:
            row = row.split()
            if row[0] in temp_name:
                index0 = name_to_index[row[0]]
            else:
                index0 = len(all_name)
                name_to_index[row[0]] = len(all_name)
                temp_name.add(row[0])
                all_name.append(row[0])

            if row[1] in temp_name:
                index1 = name_to_index[row[1]]
            else:
                index1 = len(all_name)
                name_to_index[row[1]] = len(all_name)
                temp_name.add(row[1])
                all_name.append(row[1])

            parent = index0
            child = index1
            try:
                hierarchy[parent].add(child)
            except KeyError:
                hierarchy[parent] = set([child])
            try:
                parent_of[child].add(parent)
            except KeyError:
                parent_of[child] = set([parent])
    return hierarchy, parent_of, all_name, name_to_index


def find_first_level(parent_of, all_name):
    first_level = set()
    number_level = {}
    for i in range(len(all_name)):
        try:
            if(parent_of[i]):
                continue
        except KeyError:
            number_level[i] = 0
            first_level.add(i)
    return first_level, number_level


def find_level(hierarchy, parent_of, all_name):
    first_level, number_level = find_first_level(parent_of, all_name)
    level = [first_level]
    max_level = 0
    for _, i in enumerate(level):
        old_level = {}
        for j in i:
            try:
                for k in hierarchy[j]:
                    prev_k = number_level[j]
                    try:
                        prev_k = number_level[k]
                    except KeyError:
                        pass
                    try:
                        if(prev_k < number_level[j] + 1):
                            old_level[prev_k].add(k)
                    except KeyError:
                        old_level[prev_k] = {k}
                    if(prev_k < number_level[j] + 1):
                        number_level[k] = number_level[j] + 1
                    if(max_level < number_level[k]):
                        max_level = number_level[k]
                        level.append({k})
                    else:
                        level[number_level[k]].add(k)
            except KeyError:
                continue
        for k, a in old_level.items():
            level[k] -= a
    if len(level[0]) == 1:
        level = level[1:]
    return level


def reindex_hierarchy(file_name):
    hierarchy, parent_of, all_name, name_to_index = create_hierarchy_structure(
        file_name)
    level = find_level(hierarchy, parent_of, all_name)
    remap = {}
    index = 0
    for l in level:
        for i in l:
            remap[i] = index
            index = index + 1

    new_hierarchy = {}
    for i in hierarchy:
        try:
            new_hierarchy[remap[i]] = set([remap[a] for a in hierarchy[i]])
        except KeyError:
            pass

    new_parent_of = {}
    for i in parent_of:
        try:
            new_parent_of[remap[i]] = set([remap[a] for a in parent_of[i]])
        except KeyError:
            pass

    new_level = np.concatenate([[0], np.cumsum([len(i) for i in level])])
    new_name_to_index = {}
    new_all_name = ['0'] * new_level[-1]
    for i in name_to_index:
        try:
            new_name_to_index[i] = remap[name_to_index[i]]
            new_all_name[remap[name_to_index[i]]] = i
        except KeyError:
            pass

    return new_hierarchy, new_parent_of, new_all_name, new_name_to_index, new_level


def load_hierarchy(file_name):
    with open('data/%s' % file_name, 'rb') as f:
        hierarchy, parent_of, all_name, name_to_index, level = pickle.load(
            f)
    return hierarchy, parent_of, all_name, name_to_index, level


def save_hierarchy(file_name, hierarchy, parent_of, all_name, name_to_index, level):
    directory = "data/%s" % "/".join(file_name.split("/")[:-1])
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open('data/%s' % file_name, 'wb') as f:
        pickle.dump([hierarchy, parent_of, all_name,
                     name_to_index, level], f)


def cutoff_label(cutoff_label, hierarchy, parent_of, all_name, name_to_index, level):
    remap = {}
    j = 0
    l = 0
    delete_each_level = np.zeros(len(level))
    for i, _ in enumerate(all_name):
        if not i in cutoff_label:
            if i >= level[l]:
                l = l + 1
            remap[i] = j
            j = j + 1
        else:
            if i >= level[l]:
                l = l + 1
            delete_each_level[l] = delete_each_level[l] + 1

    # process new level
    level = remap_level(level, delete_each_level)
    hierarchy = remap_hie_and_parent(hierarchy, remap)
    parent_of = remap_hie_and_parent(parent_of, remap)
    all_name = remap_all_name(all_name, remap)
    name_to_index = remap_name_to_index(all_name)
    return hierarchy, parent_of, all_name, name_to_index, level, remap


def remap_level(level, delete_each_level):
    delete_each_level = np.cumsum(delete_each_level)
    new_level = np.array(level) - delete_each_level
    last_one = new_level[-1]
    c = 0
    for i in range(len(new_level) - 1):
        if new_level[-2 - i] != last_one:
            c = len(new_level) - i
            break
    new_level = new_level[:c]
    return new_level.astype(int).tolist()


def remap_hie_and_parent(hobject, remap):
    new_hobject = {}
    for key in hobject:
        value = hobject[key]
        try:
            new_key = remap[key]
            for v in value:
                try:
                    new_v = remap[v]
                except KeyError:
                    continue
                try:
                    new_hobject[new_key].add(new_v)
                except KeyError:
                    new_hobject[new_key] = set([new_v])
        except KeyError:
            continue
    return new_hobject


def remap_all_name(all_name, remap):
    all_in = list(remap.keys())
    all_name = np.array(all_name)[all_in]
    return all_name.tolist()


def remap_name_to_index(all_name):
    new_name_to_index = {}
    for i, n in enumerate(all_name):
        new_name_to_index[n] = i
    return new_name_to_index


def save_new_hierarchy(hierarchy_name, cutoff):
    hierarchy, parent_of, all_name, name_to_index, level = load_hierarchy(
        hierarchy_name)
    hierarchy, parent_of, all_name, name_to_index, level, remap = cutoff_label(
        cutoff, hierarchy, parent_of, all_name, name_to_index, level)
    save_hierarchy(hierarchy_name, hierarchy, parent_of,
                   all_name, name_to_index, level)
    return remap
