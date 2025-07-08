# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np


def test_tensor():
    """
    多维矩阵相乘的条件：最后两个维度符合维度相乘的条件，且前面的维度符合 维度广播 的条件。
    维度广播的条件：对应维度相同，或者其中一个维度的值为1，或者该维度直接就不存在 如下例子所示。
    """
    a = torch.randn(2, 3, 7, 8)
    b = torch.randn(3, 8, 4)
    try:
        c = torch.matmul(a, b)
        print(c.shape)
    except Exception as e:
        print("矩阵不能相乘")


def test_conv():
    inputs = torch.randn(1, 3, 224, 224) # shape: [batch, channel, H, W]
    conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    outputs = conv(inputs)
    print(outputs.shape)  # [1, 64, 112, 112]


def test_layer_norm():
    batch, sentence_length, embedding_dim = 20, 5, 10
    embedding = torch.randn(batch, sentence_length, embedding_dim)
    layer_norm = nn.LayerNorm(embedding_dim)
    # Activate module
    output = layer_norm(embedding)
    print(output.shape)


def test_pool():
    # # target output size of 5x7
    # m = nn.AdaptiveAvgPool2d((5, 7))
    # input = torch.randn(1, 64, 8, 9)
    # output = m(input)

    # # target output size of 7x7 (square)
    # m = nn.AdaptiveAvgPool2d(7)
    # input = torch.randn(1, 64, 10, 9)
    # output = m(input)

    # target output size of 10x7
    m = nn.AdaptiveAvgPool2d((None, 7))
    input = torch.randn(1, 64, 10, 9)
    output = m(input)

    print(output.shape)


def test_dim_exchange():
    arr = torch.randn(2, 3, 4)
    print(type(arr))
    arr1 = arr.transpose(0, 1)  # 只能交换两个轴
    print(f"交换前shape:{arr.shape}, 交换后shape:{arr1.shape}")
    arr2 = Tensor.permute(arr, [2, 0, 1])  # 可以交换多个轴
    print(f"交换前shape:{arr.shape}, 交换后shape:{arr2.shape}")


def test_dim_operate():
    x = torch.randn(2, 3)
    y = torch.cat((x, x, x), 0)
    z = torch.cat((x, x, x), 1)
    print(f"torch.cat操作，y shape{y.shape}, z shape:{z.shape}")

    a = torch.tensor(np.arange(6).reshape(2, 3))
    b = torch.tensor(np.arange(6).reshape(2, 3))
    c = torch.stack([a, b], dim=1)
    print(f"torch.stack操作，c shape{c.shape}")
    print(c)



if __name__ == '__main__':
    # test_tensor() # 矩阵相乘
    # test_conv()  # 卷积
    # test_layer_norm() # 归一化
    # test_pool() # 池化
    # test_dim_exchange()
    test_dim_operate()

