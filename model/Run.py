#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/11/21 20:10
# @Author  : Alex
# @Site    : 
# @File    : Run.py
# @Software: PyCharm

import argparse

from ModelHandel import ModelHandel


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', '--learning_ratio', type=float, default=0.001, help='learning ratio of model')  # 学习率
    parser.add_argument('-dim', '--output_dim', type=int, default=32, help='number of hidden units in encoder')  # 表征长度
    parser.add_argument('-bs', '--batch_size', type=int, default=32, help='number of bath size')  # 训练size的大小
    parser.add_argument('-gpu', '--GPU', type=int, default=0, help='GPU ID')  # GPU编号
    parser.add_argument('-ep', '--epoch', type=int, default=20, help='number of epoch')  # 训练轮次的设置

    parser.add_argument('-trp', '--train_path', type=str, default="../config/train.csv",
                        help='training data path')
    parser.add_argument('-tep', '--test_path', type=str, default="../config/test.csv",
                        help='test data path')
    parser.add_argument('-vap', '--val_path', type=str, default="../config/val.csv",
                        help='val data path')
    parser.add_argument('-m', '--model', type=str, default="train", help='style of train')  # 选择训练还是验证
    parser.add_argument('-chp', '--check_point', type=bool, default=False, help='Whether to continue training')  # 断点设置

    args = parser.parse_args()

    # 超参设置
    train_path = args.train_path
    dim = args.output_dim
    test_path = args.test_path
    val_path = args.val_path
    batch_size = args.batch_size
    epoch = args.epoch
    gpu = args.GPU
    model = args.model
    check_point = args.check_point
    lr = args.learning_ratio

    print(args)
    bl = ModelHandel(train_path, val_path, test_path, batch_size, epoch, gpu, model, dim, lr,
                     check_point)
    if model == 'train':
        bl.train()
    elif model == 'test':
        bl.test()


if __name__ == '__main__':
    run()

#
