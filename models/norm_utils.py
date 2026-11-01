"""可训练解码器的归一化层工具（Q107）。

四次 run（ep6/7/8/9/19/20）实测：batch=2 下 BatchNorm 的 batch 统计
方差大、running stats 无界膨胀（up2 var 涨至 7.7e7，4 个数量级），激活
幅值随训练无界爬升（47→1.8e4），权重轨迹迟早进入激活溢出临界区
（pred=nan、同批重跑有限的非确定性翻转）。GroupNorm 每样本组内归一化：
无 running stats（根除膨胀）、对 batch 大小不敏感（batch=2 稳定）、
分割网络标准实践。冻结主干（Hiera/DINOv2 内部 LayerNorm）不动。
组数取 gcd(32, C) 保证整除且尽量大。
"""

import math

import torch.nn as nn


def gn(channels):
    """BatchNorm2d(channels) 的数值稳定替代：GroupNorm(gcd(32,C), C)。"""
    return nn.GroupNorm(math.gcd(32, channels), channels)