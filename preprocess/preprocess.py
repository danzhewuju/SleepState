#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/11/20 10:55 上午
# @Author  : Alex
# @Site    : 
# @File    : preprocess.py
# @Software: PyCharm

import pickle

from util.seeg_utils import *


def read_data():
    for i in range(1, 10):
        path = '../dataset/insomnia/ins{}.edf'.format(i)
        data = read_edf_raw(path)
        print("Start time:" + data.annotations.orig_time.strftime("%H:%M:%S"))
        print(get_sampling_hz(data))


def read_st():
    path = '../dataset/insomnia/ins2.edf.st'
    data = pickle.load(open(path, 'r'))
    print("yes")


read_data()
