import sys, os
import random

from torch.utils.data import Dataset, DataLoader
import torch


class Dictionary():
    def __init__(self, filename):
        self.idx2obj = []
        self.obj2idx = {}
        with open(filename) as fin:
            for i, line in enumerate(fin):
                self.obj2idx[line.strip()] = len(self.obj2idx)
                self.idx2obj.append(line.strip())

class LMdata(Dataset):
    def __init__(self, filelist, dictionaries, split):
        '''Load data_file'''
        self.data = []
        self.split = split
        with open(filelist, 'r') as f:
            lines = f.readlines()
        for line in lines:
            lineelems = line.strip().split(',')
            datapiece = {'label': int(lineelems[0]), 'feature': lineelems[1:]}
            self.data.append(datapiece)
        self.dictionaries = dictionaries

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        datapiece = self.data[idx]
        featurevec = []
        for i, feature in enumerate(datapiece['feature']):
            featurevec.append(self.dictionaries[i].obj2idx[feature])
        return featurevec, datapiece['label']

def collate_fn(batch):
    features = [feat[0] for feat in batch]
    labels = [feat[1] for feat in batch]
    return torch.LongTensor(features), torch.LongTensor(labels)

def create(datapath, dictionary=None, batchSize=1, shuffle=True, workers=0):
    loaders = []
    datatype_dict = os.path.join(datapath, 'datatypes.txt')
    recipient_dict = os.path.join(datapath, 'reciptypes.txt')
    condition_dict = os.path.join(datapath, 'conditions.txt')
    datatype_dict = Dictionary(datatype_dict)
    recipient_dict = Dictionary(recipient_dict)
    condition_dict = Dictionary(condition_dict)
    dictionaries = [datatype_dict, recipient_dict, condition_dict]
    for split in ['train', 'valid', 'test']:
        data_file = os.path.join(datapath, '%s.csv' %split)
        dataset = LMdata(data_file, dictionaries, split)
        loaders.append(DataLoader(dataset=dataset, batch_size=batchSize,
                                  shuffle=shuffle, collate_fn=collate_fn,
                                  num_workers=workers))
    return loaders[0], loaders[1], loaders[2], datatype_dict, recipient_dict, condition_dict
