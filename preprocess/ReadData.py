#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/11/21 14:01
# @Author  : Alex
# @Site    : 
# @File    : ReadData.py
# @Software: PyCharm


"""
数据预处理的相关类，主要实现对于数据的采集已经处理方式
"""

import sys

from tqdm import tqdm

sys.path.append('../')

from util.seeg_utils import *


class PdfProcess:
    def create_split_data(self, save_dir='../dataset/preprocessedData', epoch=30, downsampling=128):
        """
        :param save_dir: 切分的数据集存放的位置 默认位置：../preprocessedData
        :param epoch: 切分的数据的大小，默认设定的大小30s

        :return:
        """

        def cal_time_index(time_a, time_b):
            """

            :param time_a: 起始时间段
            :param time_b: 终止时间段
            :return: 计算在绝对的时间下的相差的秒数
            """
            int_a = [int(x) for x in time_a.split(':')]
            int_b = [int(x) for x in time_b.split(':')]
            d_s = int_b[2] - int_a[2]
            if d_s < 0:
                d_s += 60
                int_b[1] -= 1
            d_m = int_b[1] - int_a[1]
            if d_m < 0:
                d_m += 60
                int_b[0] -= 1
            d_h = int_b[0] - int_a[0]
            d_h = 24 + d_h if d_h < 0 else d_h
            return d_h * 3600 + d_m * 60 + d_s

        # 如果文件夹不存在，需要自动创建,同时创建睡眠分期的文件夹
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
            print("{} 文件夹不存在，已经自动创建了文件夹！".format(save_dir))
            for d in self.sleepState.keys():
                os.mkdir(os.path.join(save_dir, d))  # 为每一个睡眠状态创建一个文件夹
        # 将每一个睡眠的睡眠状态映射为字典路径
        sleep_state_path = dict()  # {'W':'../dataset/preprocessedData/W}
        for state in os.listdir(save_dir):
            sleep_state_path[state] = os.path.join(save_dir, state)

        # 读取相关数据集的文件夹，进行数据切分操作
        # 1. 所有文件包含的全部路径
        all_file = [os.path.join(self.path_dir, x) for x in os.listdir(self.path_dir)]
        raw_data = list(filter(lambda x: x.split('.')[-1] == "edf", all_file))  # 过滤掉其他不是原文件的文件
        raw_data.sort()  # 按照顺序处理
        # 每一个原文件对讲的标签
        data_label = dict()
        for path in raw_data:
            path_label = path[:-3] + "txt"
            data_label[path] = path_label

        # 2. 开始做数据的切分
        for path_data in tqdm(raw_data):
            path_data_label = data_label[path_data]
            label_info = pd.read_csv(path_data_label, sep='\t', skiprows=range(0, 21))
            data_split_length = label_info['Duration[s]'].tolist()  # 单个时间的长度
            # 获得数据的长度，同时去掉那些发作时长不足30s的数据
            state = [x for i, x in enumerate(label_info["Sleep Stage"].tolist()) if
                     data_split_length[i] == epoch]  # 睡眠的状态,
            time = [x for i, x in enumerate(label_info['Time [hh:mm:ss]'].tolist()) if
                    data_split_length[i] == epoch]  # 睡眠的时间

            # 需要将标准的时间会序列中的序号

            data = read_edf_raw(path_data)  # 读取原始的数据，此时的数据可能比较占用内存，内存占用大概为2GB
            data.resample(downsampling, npad='auto')  # 对数据进行降采样，降低对于显卡的占用大小
            sampling = get_sampling_hz(data)  # 获得数据的采样频率
            start_time_absolute = data.annotations.orig_time.strftime("%H:%M:%S")  # 获得文件的绝对起始时间
            for s, t in tqdm(zip(state, time)):
                start_time_file = cal_time_index(start_time_absolute, t)  # 计算相对稳健的起始时间
                # 获得了数据的切片
                split_data, _ = data[:, start_time_file * downsampling: (start_time_file + epoch) * downsampling]
                # 设置数据的存储目录
                name = "{}_{}_{}.{}".format(uuid.uuid1(), t, s, "npy")  # 命名规则：uuid+时间+状态+后缀
                save_split_data_path = os.path.join(sleep_state_path[s], name)  # 完整的路基表示
                save_numpy_info(split_data, save_split_data_path)  # 写入到本地的磁盘中

    def __init__(self, path_dir):
        """
        path_dir: 数据存放的文件夹
        """
        self.path_dir = path_dir
        # 睡眠状态分为5个时期分别为：W, S1, S2, S3, R 这5个状态，对应的的标签分别为 0-4
        self.sleepState = {"W": 0, "S1": 1, "S2": 2, "S3": 3, "R": 4}
        # self.create_split_data()  # 根据默认的参数来生成切片的数据