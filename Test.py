# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Real Flow 

# %%
from data import Dataset
from embedding import Doc2Vec, NoTag_Doc2Vec, OnlyLeafDoc2Vec
from assemble_classifier import ESLNN, SHLNN, HMC_LMLP, LCPL
import shutil
import os
import numpy as np
import json


# %%
evaluater_config = json.load(open('evaluater_config.json'))
data_name = evaluater_config['train_model_folder']
test_file_name = evaluater_config['file_name']


# %%
config = json.load(open('export/%s/model_detail.json' % data_name))
classification_type = config['classification_type']
correction = config['correction']
mandatory_leaf = config['mandatory_leaf']
hidden = config['hidden']
target_hidden = config['target_hidden']
embedding_size = config['embedding_size']
embedding_type = "OPD"
hidden = config['hidden']
target_hidden = config['target_hidden']
threshold = config['threshold']


# %%
print("---------------Preprocessing-----------------")


# %%
if classification_type == "multi-class":
    mandatory_leaf = True
elif classification_type == "multi-label":
    mandatory_leaf = False


# %%
if os.path.isdir('data/%s/output' % data_name):
    shutil.rmtree('data/%s/output' % data_name)
if os.path.isdir('data/%s/store' % data_name):
    shutil.rmtree('data/%s/store' % data_name)


# %%
dataset_test = Dataset(data_name, "test", test_split=False, classification_type=classification_type, data_file_name=test_file_name)


# %%
print("---------------Document embedding-----------------")


# %%
if embedding_type == "LOD":
    doc2vec = OnlyLeafDoc2Vec(data_name, dataset_test.number_of_classes(), size=embedding_size, epoch=270, batch_size=10000)
elif embedding_type == "Normal":
    doc2vec = NoTag_Doc2Vec(data_name, dataset_test.number_of_classes(), size=embedding_size, epoch=270, batch_size=10000)
else:
    doc2vec = Doc2Vec(data_name, dataset_test.number_of_classes(), size=embedding_size, epoch=270, batch_size=10000)
doc2vec.load_model('export/%s/doc2vec.model' % data_name)


# %%
dataset_test.change_to_Doc2Vec(doc2vec)


# %%
print("---------------Training classifiers-----------------")


# %%
model = ESLNN(data_name, dataset_test, "temp", dataset_test, iteration=2000, stopping_time=300, batch_size=65536, hidden_size=hidden, target_hidden_size=target_hidden, use_dropout=True, start_level=99999)


# %%
model.train()
model.apply_threshold(threshold)


# %%
f = open('export/%s/result_%s' % (data_name, test_file_name), 'w')


# %%
print("---------------Evaluation-----------------")


# %%
mode = 'test'
model.export_result(mode, correction=correction, mandatory_leaf=mandatory_leaf, file_name=test_file_name)
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

