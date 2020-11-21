#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/11/20 10:55 上午
# @Author  : Alex
# @Site    : 
# @File    : preprocess.py
# @Software: PyCharm

from util.seeg_utils import *
import pickle


def read_data():
    path = '../dataset/insomnia/ins2.edf.st'
    data = read_edf_raw(path)
    print("Yes")


def read_st():
    path = '../dataset/insomnia/ins2.edf.st'
    data = pickle.load(open(path, 'r'))
    print("yes")


read_st()
