# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Real Flow 

# %%
from data import Dataset
from embedding import Doc2Vec, NoTag_Doc2Vec, OnlyLeafDoc2Vec
from assemble_classifier import ESLNN
import shutil
import os
import numpy as np
import json

# %%
config = json.load(open('config.json'))
data_name = config['data_name']
train_file_name = config['train_file_name']
test_file_name = config['test_file_name']
classification_type = config['classification_type']
test_split = config['test_split']
predict_test = config['predict_test']
evaluate_test = config['evaluate_test']
correction = config['correction']
mandatory_leaf = config['mandatory_leaf']
hidden = config['hidden']
target_hidden = config['target_hidden']
embedding_size = config['embedding_size']
embedding_type = 'OPD'


# %%
print("---------------Preprocessing-----------------")


# %%
if classification_type == "multi-class":
    mandatory_leaf = True
elif classification_type == "multi-label":
    mandatory_leaf = False


# %%
if not os.path.isdir('export/%s' % data_name):
    os.makedirs('export/%s' % data_name)


# %%
if os.path.isdir('data/%s/output' % data_name):
    shutil.rmtree('data/%s/output' % data_name)
if os.path.isdir('data/%s/store' % data_name):
    shutil.rmtree('data/%s/store' % data_name)


# %%
dataset_train = Dataset(data_name, "train", test_split=test_split, classification_type=classification_type, data_file_name=train_file_name)
dataset_validate = Dataset(data_name, "validate", test_split=test_split, classification_type=classification_type, data_file_name=train_file_name)
if (test_split or predict_test or evaluate_test):
    dataset_test = Dataset(data_name, "test", test_split=test_split, classification_type=classification_type, data_file_name=test_file_name)
else:
    dataset_test = "temp"


# %%
print("---------------Training document embedding-----------------")


# %%
if embedding_type == "LOD":
    doc2vec = OnlyLeafDoc2Vec(data_name, dataset_train.number_of_classes(), size=embedding_size, epoch=270, batch_size=10000)
elif embedding_type == "Normal":
    doc2vec = NoTag_Doc2Vec(data_name, dataset_train.number_of_classes(), size=embedding_size, epoch=270, batch_size=10000)
else:
    doc2vec = Doc2Vec(data_name, dataset_train.number_of_classes(), size=embedding_size, epoch=270, batch_size=10000)
doc2vec.fit(dataset_train.datas, dataset_train.labels, dataset_validate.datas, dataset_validate.labels, early_stopping=False)
# doc2vec.load_model('export/%s/doc2vec.model' % data_name)


# %%
dataset_train.change_to_Doc2Vec(doc2vec)
dataset_validate.change_to_Doc2Vec(doc2vec)
if (test_split or predict_test or evaluate_test):
    dataset_test.change_to_Doc2Vec(doc2vec)


# %%
if hidden == 'auto' or target_hidden == 'auto':
    a = []
    for i in range(len(dataset_train.level)-1):
        a.append(dataset_train.level[i+1] - dataset_train.level[i])
    a = np.array(a)

    if hidden == 'auto':
        hidden = a*2 + 300
        hidden[hidden > 3000] = 3000
        hidden = hidden.tolist()
    if target_hidden == 'auto':
        target_hidden = a[:-1]*2 + 30
        target_hidden[target_hidden > 100] = 100
        target_hidden = target_hidden.tolist()


# %%
print("---------------Training classifiers-----------------")


# %%
model = ESLNN(data_name, dataset_train, dataset_validate, dataset_test, iteration=2000, stopping_time=300, batch_size=65536, hidden_size=hidden, target_hidden_size=target_hidden, use_dropout=True, start_level=0)


# %%
model.train()


# %%
threshold = model.tuning_threshold()


# %%
f = open('export/%s/result.txt' % data_name, 'w')


# %%
print("---------------Evaluation-----------------")


# %%
list_of_mode = ['train', 'validate']
if (test_split or predict_test or evaluate_test):
    list_of_mode.append('test')


# %%
for mode in list_of_mode:
    if predict_test or mode != 'test':
        model.export_result(mode, correction=correction, mandatory_leaf=mandatory_leaf)
    if evaluate_test or mode != 'test':
        f1_macro, f1_micro, f1_each = model.evaluate(mode, correction=correction, mandatory_leaf=mandatory_leaf)
        f.write("--------------------------- %s -------------------------------\n" % mode)
        print("--------------------------- %s -------------------------------" % mode)
        f.write("F1 macro: %.4f F1 micro: %.4f\n" % (f1_macro, f1_micro))
        print("F1 macro: %.4f F1 micro: %.4f" % (f1_macro, f1_micro))
        if classification_type == 'hierarchical':
            for level, (macro, micro) in enumerate(f1_each):
                f.write("Level: %d F1 macro: %.4f F1 micro: %.4f\n" % (level, macro, micro))
                print("Level: %d F1 macro: %.4f F1 micro: %.4f" % (level, macro, micro))
            f.write('\n')
            print('')


# %%
f.close()


# %%



# %%
config['hidden'] =  hidden
config['target_hidden'] = target_hidden
config['threshold'] = threshold
with open('export/%s/model_detail.json' % data_name, 'w') as f:
    json.dump(config, f)

