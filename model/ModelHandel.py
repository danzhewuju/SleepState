#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/11/21 20:17
# @Author  : Alex
# @Site    : 
# @File    : modelHandel.py
# @Software: PyCharm
import os
import time

import torch
import torch.nn as nn
from tqdm import tqdm

from data_util import MyData
from myModel import RCNN
from example.util_file import IndicatorCalculation


class ModelHandel:
    def __init__(self, train_path, val_path, test_path, batch_size, epoch, gpu, model, out_put_size, learning_ratio,
                 check_point):
        """

        :param train_path: 训练数据集的路径
        :param val_path:  验证数据的路径
        :param test_path: 测试数据集的路径
        :param batch_size: batch size 的大小
        :param epoch: 训练的轮次
        :param gpu:  指定GPU运行
        :param model: 训练还是验证
        :param check_point: 是否从断点开始验证
        """
        self.epoch = epoch  # 迭代次数
        self.batch_size = batch_size  # 批次的大小
        self.dim = out_put_size  # 数据的维度
        self.lr = learning_ratio  # 学习率
        self.train_path = train_path  # 训练的数据集
        self.test_path = test_path  # 测试的训练集
        self.val_path = val_path  # 验证集
        self.m = model  # 训练集和测试的选择
        self.gpu = gpu  # gpu选择
        self.model = RCNN(gpu, out_put_size, Resampling=128).cuda(gpu) if gpu >= 0 else RCNN(gpu, out_put_size,
                                                                                             Resampling=128)
        if check_point:
            self.load_model()  # 如果是断点训练
            print(" Start checkpoint training")

        pass

    def save_mode(self, save_path='../save_model'):
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        save_full_path = os.path.join(save_path, 'RCNN.pkl')
        torch.save(self.model.state_dict(), save_full_path)
        print("Saving Model in {}......".format(save_full_path))
        return

    def load_model(self, model_path='../save_model/RCNN.pkl'):
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
            print("Loading Baseline Mode from {}".format(model_path))
        else:
            print("Model is not exist in {}".format(model_path))
            exit()
        return

    @staticmethod
    def log_write(result, path='../log/log.txt'):
        time_stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        dir_name = os.path.dirname(path)
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        if not os.path.exists(path):
            f = open(path, 'w')
        else:
            f = open(path, 'a')
        result_log = result + "\t" + time_stamp + '\n'
        print(result_log)
        f.write(result_log)
        f.close()
        print("Generating log!")
        return

    def evaluation(self, prey, y):
        '''
        评价指标的计算
        :param y:    实际的结果
        :return:  返回各个指标是的结果
        '''
        result = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1score': 0}

        cal = IndicatorCalculation(prey, y)
        result['accuracy'] = cal.get_accuracy()
        result['precision'] = cal.get_precision()
        result['recall'] = cal.get_recall()
        result['f1score'] = cal.get_f1score()

        return result

    def train(self):
        mydata = MyData(path_train=self.train_path, path_test=self.test_path, path_val=self.val_path,
                        batch_size=self.batch_size)

        train_data_loader = mydata.data_loader(None, mode='train')
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_func = nn.CrossEntropyLoss()

        acc_train, loss_train = [], []
        last_test_accuracy = 0
        with tqdm(total=self.epoch * len(train_data_loader)) as tq:

            for epoch in tqdm(range(self.epoch)):
                for step, (b_x, b_y) in enumerate(tqdm(train_data_loader)):  # gives batch data
                    b_x_g = b_x.cuda(self.gpu)

                    b_y_g = b_y.cuda(self.gpu)
                    output = self.model(b_x_g)  #
                    loss = loss_func(output, b_y_g)  # cross entropy loss

                    optimizer.zero_grad()  # clear gradients for this training step
                    loss.backward()  # backpropagation, compute gradients
                    optimizer.step()  # apply gradients
                    pred_y = torch.max(output, 1)[1].data
                    res_tmp = [1 if pred_y[i] == b_y[i] else 0 for i in range(len(b_y))]
                    acc_train += res_tmp
                    loss_train.append(loss.data.cpu())
                    if step % 10 == 0:
                        acc_test, loss_test = [], []
                        for x_test, label_test in next(mydata.next_batch_val_data(transform=None)):
                            # x_test = linear_matrix_normalization(x_test)
                            if self.gpu >= 0:
                                x_test, label_test = x_test.cuda(self.gpu), label_test.cuda(self.gpu)
                            with torch.no_grad():
                                label_output_test = self.model(x_test)
                                loss_label = loss_func(label_output_test, label_test)
                                pre_y_test = torch.max(label_output_test, 1)[1].data
                                acc_test += [1 if pre_y_test[i] == label_test[i] else 0 for i in range(len(label_test))]
                                loss_test.append(loss_label.data.cpu())
                        acc_train_avg = sum(acc_train) / len(acc_train)
                        loss_train_avg = sum(loss_train) / len(loss_train)

                        acc_test_avg = sum(acc_test) / len(acc_test)
                        loss_test_avg = sum(loss_test) / len(loss_test)

                        print(
                            'Epoch:{} | Step:{} | train loss:{:.6f} | val loss:{:.6f} | train accuracy:{:.5f} | val accuracy:{:.5f}'.format(
                                epoch, step, loss_train_avg, loss_test_avg, acc_train_avg, acc_test_avg))
                        acc_train.clear()
                        loss_train.clear()
                        if last_test_accuracy == 1:
                            last_test_accuracy = 0
                        if last_test_accuracy <= acc_test_avg:
                            self.save_mode()  # 保存较好的模型
                            print("Saving model...")
                            last_test_accuracy = acc_test_avg
                    tq.update(1)
        return

    def test(self):
        self.load_model()  # 加载模型
        mydata = MyData(path_train=self.train_path, path_test=self.test_path, path_val=self.val_path,
                        batch_size=self.batch_size)
        test_data_loader = mydata.data_loader(mode='test', transform=None)
        acc = []
        loss = []

        grand_true = []
        prediction = []
        probability = []

        loss_func = nn.CrossEntropyLoss()
        for step, (x, label) in enumerate(tqdm(test_data_loader)):
            if self.gpu >= 0:
                x, label = x.cuda(self.gpu), label.cuda(self.gpu)
            with torch.no_grad():
                label_output = self.model(x)
                loss_test = loss_func(label_output, label)
                loss_total = loss_test
                prey = torch.max(label_output, 1)[1].data

                y = label.cpu()
                acc += [1 if prey[i] == y[i] else 0 for i in range(len(y))]
                loss.append(loss_total.data.cpu())
                # if recoding:
                # ids_list += ids
                grand_true += [int(x) for x in y]
                prediction += [int(x) for x in prey]
                # probability += [float(x) for x in torch.softmax(label_output, dim=1)[:, 1]]
        loss_avg = sum(loss) / len(loss)
        accuracy = sum(acc) / len(acc)
        result = "|Data size:{}| test loss:{:.6f}| Accuracy:{:.5f} ".format(len(acc), loss_avg, accuracy)
        ModelHandel.log_write(result)
