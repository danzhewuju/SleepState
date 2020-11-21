#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/11/21 16:04
# @Author  : Alex
# @Site    : 
# @File    : Run.py
# @Software: PyCharm
import argparse
from functools import partial

from ReadData import PdfProcess


def set_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--dataProcessing', default='dataSplit', type=str, help="Processing Data by your choice")
    return parser


def run():
    arg = set_parser().parse_args()
    pdfProcess = PdfProcess('../dataset/insomnia')
    op = {'dataSplit': partial(pdfProcess.create_split_data),
          'createDataset': partial(pdfProcess.create_data_set)}  # 设置操作的集合
    op[arg.dataProcessing]()


run()
