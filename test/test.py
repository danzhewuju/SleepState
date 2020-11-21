#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/11/21 14:18
# @Author  : Alex
# @Site    : 
# @File    : test.py
# @Software: PyCharm

# path = "/data/yh/Python/SleepState/dataset/insomnia/ins1.txt"
# data = pd.read_csv(path, sep='\t', skiprows=range(0, 21))
# print(data)
# state = data["Sleep Stage"]
# Time = data['Time [hh:mm:ss]']
# print(state)
# print(Time)
import uuid


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


print(cal_time_index("23:36:48", "00:22:48"))


print(uuid.uuid1())
