# -*- coding: utf-8 -*-
# @Time    : 2020/11/26 15:47
# @Author  : liuwei
# @File    : vocab.py

"""
item2id vocab

for word

for labels

for boundary

"""

import os
import json
import pickle

class ItemVocabFile():
    """
    Build vocab from file.
    Note, each line is a item in vocab, or each items[0] is in vocab
    根据从真实标签文件(labels.txt)来制作包含id序列的词典。
    主要的做法：
        1.制作id序列的字典文件。
        2.添加'<pad>','<unk>'等特殊字符进去。
    Args:
        files(list): 包含所有标签文件的路径，外面套了一层[]，此处为labels.txt,包含了遇到实验中的所有标签。
    """
    def __init__(self, files, is_word=False, has_default=False, unk_num=0):
        self.files = files
        self.item2idx = {}
        self.idx2item = []
        self.item_size = 0
        self.is_word = is_word
        if not has_default and self.is_word:
            self.item2idx['<pad>'] = self.item_size
            self.idx2item.append('<pad>')
            self.item_size += 1
            self.item2idx['<unk>'] = self.item_size
            self.idx2item.append('<unk>')
            self.item_size += 1
            # for unk words
            for i in range(unk_num):
                self.item2idx['<unk>{}'.format(i+1)] = self.item_size
                self.idx2item.append('<unk>{}'.format(i+1))
                self.item_size += 1

        self.init_vocab()

    def init_vocab(self):
        for file in self.files:
            with open(file, "rb") as f:
                class2idx = pickle.load(f)
            self.item2idx=class2idx
            for key in class2idx.keys():
                self.idx2item.append(key)
            self.item_size=len(self.idx2item)
        
    def get_item_size(self):
        return self.item_size

    def convert_item_to_id(self, item):
        if item in self.item2idx:
            return self.item2idx[item]
        elif self.is_word:
            unk = "<unk>" + str(len(item))
            if unk in self.item2idx:
                return self.item2idx[unk]
            else:
                return self.item2idx['<unk>']
        else:
            print("Label does not exist!!!!")
            print(item)
            raise KeyError()

    def convert_items_to_ids(self, items):
        return [self.convert_item_to_id(item) for item in items]

    def convert_id_to_item(self, id):
        return self.idx2item[id]

    def convert_ids_to_items(self, ids):
        return [self.convert_id_to_item(id) for id in ids]

class ItemVocabArray():
    """
    Build vocab from file.
    根据从词汇树中获取到的matched_words来制作包含id序列的词典。
    主要的做法：
        1.制作id序列的字典文件。
        2.添加'<pad>','<unk>'等特殊字符进去。

    Note, each line is a item in vocab, or each items[0] is in vocab
    Args:
        items_array(list): 从词汇树中获取到的matched_words。

    """
    def __init__(self, items_array, is_word=False, has_default=False, unk_num=0):
        self.items_array = items_array
        self.item2idx = {}
        self.idx2item = []
        self.item_size = 0
        self.is_word = is_word
        if not has_default and self.is_word:
            self.item2idx['<pad>'] = self.item_size
            self.idx2item.append('<pad>')
            self.item_size += 1
            self.item2idx['<unk>'] = self.item_size
            self.idx2item.append('<unk>')
            self.item_size += 1
            # for unk words
            for i in range(1, unk_num+1):
                self.item2idx['<unk>{}'.format(i+1)] = self.item_size
                self.idx2item.append('<unk>{}'.format(i+1))
                self.item_size += 1

        self.init_vocab()

    def init_vocab(self):
        for item in self.items_array:
            self.item2idx[item] = self.item_size
            self.idx2item.append(item)
            self.item_size += 1

    def get_item_size(self):
        return self.item_size

    def convert_item_to_id(self, item):
        if item in self.item2idx:
            return self.item2idx[item]
        elif self.is_word:
            unk = "<unk>" + str(len(item))
            if unk in self.item2idx:
                return self.item2idx[unk]
            else:
                return self.item2idx['<unk>']
        else:
            print("Label does not exist!!!!")
            print(item)
            raise KeyError()

    def convert_items_to_ids(self, items):
        return [self.convert_item_to_id(item) for item in items]

    def convert_id_to_item(self, id):
        return self.idx2item[id]

    def convert_ids_to_items(self, ids):
        return [self.convert_id_to_item(id) for id in ids]
