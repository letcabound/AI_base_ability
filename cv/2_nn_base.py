# -*- coding: utf-8 -*-
import torch


'''
激活函数（）
损失函数选择
优化器选择
  学习率
'''

def nn_demo():
    """
    1. 数据准备：输入数据 + label
    2. 网络结构搭建：激活函数 + 损失函数 + 权重初始化
    3. 优化器选择
    4. 训练策略：学习率 梯度权重 正则化
    :return:
    """
    intput = torch.tensor([5, 10]).reshape(1, 2).to(torch.float32)
    label = torch.tensor([0.01, 0.99]).reshape(1, 2)

    linear_1 = torch.nn.Linear(2, 3)
    act_1 = torch.nn.Sigmoid()  # 激活函数是 无状态(stateless)模块，即：内部没有可训练的参数。所以，该模块 只定义一次并在forward中重复使用。
    linear_2 = torch.nn.Linear(3, 2)

    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.SGD([{"params": linear_1.parameters()},{"params": linear_2.parameters()}], lr=0.5)


    for i in range(100):
        optimizer.zero_grad() # 梯度清空
        x = linear_1(intput)
        x = act_1(x)
        x = linear_2(x)
        out = act_1(x)
        loss = loss_func(out, label) # 计算损失
        print(f"第{i}次训练，损失为{loss}")
        loss.backward() # 反向梯度累积计算
        optimizer.step() # 梯度更新

def nn_demo_2():
    pass


if __name__ == '__main__':
    nn_demo()
