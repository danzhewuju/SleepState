import random
import sys

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

sys.path.append('../')
from util.util_file import matrix_normalization
import torch


class SingleDataInfo:
    """
    单个文件的存储读取，用于验证
    """

    def __init__(self, data, label, data_length):
        """
        :param file_path: 文件的名称
        :param label:  文件的标签
        :param data_length: 设置的数据长度
        """
        self.dict_label = {"pre_seizure": 1, "non_seizure": 0}
        self.input = data  # 数据源
        self.label = self.dict_label[label]  # 数据标签
        self.resampling = 500  # 系统自带的采样频率
        length = self.input.shape[-1]  # 数据的长度
        time_info = []  # 用保存截取数据的时间位置信息
        for i in range(length // (self.resampling * data_length)):
            start, end = i * self.resampling * data_length, (i + 1) * self.resampling * data_length
            time_info.append((start, end))
        self.time_info = time_info


class SingleDataset(Dataset):
    # 重写单个样本的 dataset
    def __init__(self, data, time_info, label):
        self.data = data
        self.time_info = time_info
        self.label = label

    def __getitem__(self, item):
        start, end = self.time_info[item][0], self.time_info[item][1]
        data = self.data[:, start:end]
        result = matrix_normalization(data, (100, -1))
        result = result.astype('float32')
        result = result[np.newaxis, :]
        return result, self.time_info[item], self.label

    def __len__(self):
        """
        数据的长度
        :return:
        """
        return len(self.time_info)


class DataInfo:
    """
    用于模型训练的分段信息
    """

    def __init__(self, path_data):
        data = pd.read_csv(path_data)
        data_path = data['id'].tolist()
        labels = data['label'].tolist()
        self.data = []
        for i in range(len(data_path)):
            self.data.append((data_path[i], int(labels[i])))
        self.data_length = len(self.data)

    def next_batch_data(self, batch_size):  # 用于返回一个batch的数据
        N = self.data_length
        start = 0
        end = batch_size
        random.shuffle(self.data)
        while end < N:
            yield self.data[start:end]
            start = end
            end += batch_size
            if end >= N:
                start = 0
                end = batch_size


class MyDataset(Dataset):  # 重写dateset的相关类
    def __init__(self, data, transform=None, target_transform=None):
        self.data = data
        self.transform_data = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        data_path, label = self.data[index]
        data = np.load(data_path)
        # 获得该数据的
        if self.transform_data:
            data = self.transform_data(data)
        result = matrix_normalization(data, (32, 128))  # 设置输入的格式问题
        result = result.astype('float32')
        result = result[np.newaxis, :]
        # result = trans_data(vae_model, result)
        return result, label

    def __len__(self):
        return len(self.data)


class MyData:
    def __init__(self, path_train=None, path_test=None, path_val=None, batch_size=16):
        """

        :param path_train: 训练集数据的路径
        :param path_test: 测试集数据的路径
        :param path_val: 验证集数据的路径
        :param batch_size: 批量的数据
        """

        self.path_train = path_train
        self.path_test = path_test
        self.path_val = path_val
        self.batch_size = batch_size

    def collate_fn(self, data):  #
        """
        用于自己构造时序数据，包含数据对齐以及数据长度
        :param data: torch dataloader 的返回形式
        :return:
        """
        # 主要是用数据的对齐
        data.sort(key=lambda x: x[0].shape[-1], reverse=True)
        max_shape = data[0][0].shape
        labels = []  # 每个数据对应的标签
        length = []  # 记录真实的数目长度
        domains = []
        ids = []  # 记录序列的 id
        for i, (d, patient, id_, label) in enumerate(data):
            reshape = d.shape
            length.append(d.shape[-1])
            if reshape[-1] < max_shape[-1]:
                tmp_d = np.pad(d, ((0, 0), (0, 0), (0, max_shape[-1] - reshape[-1])), 'constant')
                data[i] = tmp_d
            else:
                data[i] = d

            labels.append(label)
            domains.append(patient)
            ids.append(id_)

        return torch.from_numpy(np.array(data)), torch.tensor(labels), torch.tensor(domains), torch.tensor(length), ids

    def data_loader(self, transform, mode='train'):  # 这里只有两个模式，一个是train/一个是test
        dataloader = None
        if mode == 'train':
            # 如果加入了少样本学习的方法，需要额外的处理
            data_info = DataInfo(self.path_train)

            dataset = MyDataset(data_info.data, transform=transform)

            # 因为样本的数目不均衡，需要进行不均衡采样
            # 需要计算每一个样本的权重值

            dataloader = DataLoader(dataset, shuffle=True, batch_size=self.batch_size)

        elif mode == 'test':  # test
            data_info = DataInfo(self.path_test)
            dataset = MyDataset(data_info.data, transform=transform)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        else:
            pass

        return dataloader

    def next_batch_val_data(self, transform):
        data_info = DataInfo(self.path_val)
        dataset = MyDataset(next(data_info.next_batch_data(self.batch_size)), transform=transform)
        next_batch_data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        yield next_batch_data_loader