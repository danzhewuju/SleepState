#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/11/21 14:01
# @Author  : Alex
# @Site    : 
# @File    : ReadData.py
# @Software: PyCharm


"""
数据预处理的相关类，主要实现对于数据处理以及训练数据划分的方法
"""
import collections
import random
import sys

from tqdm import tqdm

sys.path.append('../')

from util.seeg_utils import *


class PdfProcess:
    def create_split_data(self, save_dir='../dataset/preprocessedData', epoch=30, downsampling=128, highPass=0,
                          lowPass=30):
        """
        :param downsampling: 下采样的频率
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
            location = [x for i, x in enumerate(label_info['Location'].tolist()) if data_split_length[i] == epoch]

            # 需要将标准的时间会序列中的序号

            data = read_edf_raw(path_data)  # 读取原始的数据，此时的数据可能比较占用内存，内存占用大概为2GB
            data = filter_hz(data, highPass, lowPass)  # 选择滤波范围
            data.resample(downsampling, npad='auto')  # 对数据进行降采样，降低对于显卡的占用大小
            channel_list = get_channels_names(data)  # 获得数据的信道列表
            sampling = get_sampling_hz(data)  # 获得数据的采样频率
            if sampling != downsampling:
                assert "采样出错！"
            start_time_absolute = data.annotations.orig_time.strftime("%H:%M:%S")  # 获得文件的绝对起始时间
            for s, t, l in tqdm(zip(state, time, location)):
                start_time_file = cal_time_index(start_time_absolute, t)  # 计算相对稳健的起始时间
                # 获得了数据的切片
                split_data, _ = data[:, start_time_file * downsampling: (start_time_file + epoch) * downsampling]
                # 设置数据的存储目录
                if l not in channel_list:
                    # 存在不存在的情况，这个情况比较特殊；可能是数据集存在问题
                    continue
                else:
                    index_of_channel = channel_list.index(l)  # 在该数据列表中的序号
                    split_data = split_data[index_of_channel]
                    file_name = os.path.basename(path_data).split('.')[0]
                    name = "{}_{}_{}_{}_ch_{}.{}".format(uuid.uuid1(), file_name, t, s, l,
                                                         "npy")  # 命名规则：uuid+文件+时间+状态+发生的信道+后缀
                    save_split_data_path = os.path.join(sleep_state_path[s], name)  # 完整的路基表示
                    if split_data.shape[-1] == epoch * downsampling:  # 需要严格保存有30s的数据，否则后面会出现长度不一致的问题，影响训练
                        save_numpy_info(split_data, save_split_data_path)  # 写入到本地的磁盘中
            print("Processing finished.")

    def create_data_set(self, save_dir="../config/"):
        """
        :param save_dir:
        :return: None
        将切分好的数据集划分为 train.csv, validation.csv, test.csv 数据集
        """
        if not os.path.exists(save_dir): os.mkdir(save_dir)  # 不存在该目录需要创建该目录
        train_ratio, validation_ratio = 0.7, 0.2  # 数据集train:val:test的划分按照：7:2:1进行划分
        data_dir = "../dataset/preprocessedData"  # 存放切片的数据集
        states = [os.path.join(data_dir, x) for x in os.listdir(data_dir)]
        train_data, validation_data, test_data = collections.defaultdict(list), collections.defaultdict(
            list), collections.defaultdict(list)
        for p in states:
            full_path = [os.path.join(p, x) for x in os.listdir(p)]
            random.shuffle(full_path)  # 打乱序号
            start = int(len(full_path) * train_ratio)
            mid = int(len(full_path) * (validation_ratio + train_ratio))
            train_data['id'] += full_path[:start]
            train_data['label'] += [self.sleepState[os.path.basename(p)]] * start
            train_data['sleepState'] += [os.path.basename(p)] * start
            validation_data['id'] += full_path[start:mid]
            validation_data['label'] += [self.sleepState[os.path.basename(p)]] * (mid - start)
            validation_data['sleepState'] += [os.path.basename(p)] * (mid - start)
            test_data['id'] += full_path[mid:]
            test_data['label'] += [self.sleepState[os.path.basename(p)]] * (len(full_path) - mid)
            test_data['sleepState'] += [os.path.basename(p)] * (len(full_path) - mid)

        train_data_frame, val_data_frame, test_data_frame = pd.DataFrame(train_data), pd.DataFrame(
            validation_data), pd.DataFrame(test_data)
        save_train_path, save_val_path, save_test_path = os.path.join(save_dir, "train.csv"), os.path.join(save_dir,
                                                                                                           "val.csv"), os.path.join(
            save_dir, "test.csv")
        train_data_frame.to_csv(save_train_path, index=False)
        val_data_frame.to_csv(save_val_path, index=False)
        test_data_frame.to_csv(save_test_path, index=False)
        print("Data has been split.")

    def __init__(self, path_dir):
        """
        path_dir: 数据存放的文件夹
        """
        self.path_dir = path_dir
        # 睡眠状态分为5个时期分别为：W, S1, S2, S3, R 这5个状态，对应的的标签分别为 0-4
        self.sleepState = {"W": 0, "S1": 1, "S2": 2, "S3": 3, "S4": 4, "R": 5}
        # self.create_split_data()  # 根据默认的参数来生成切片的数据
