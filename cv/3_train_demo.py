# -*- coding: utf-8 -*-
import argparse
from logging import critical

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda import device
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

"""
1. 命令行解析参数
2. device设置
3. 数据准备
4. 模型搭建
5. 优化器及学习率配置
6. 模型持久化存储
"""

"""
QA:
1. 该分类模型的损失函数 CrossEntropyLoss，为什么训练的时候用默认参数mean，测试的时候要设置 reduction='sum'？
    在训练时损失的reduction参数设置为 mean，保证不同batch_size时的反向传播的梯度尺度一致，过程不受batch_size大小影响，稳定学习率和训练过程；
    在测试时用 sum，是为了对整个测试集的损失求和，然后计算整个测试集的平均损失，便于评估。
2. other
"""

# 模型搭建
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 5, 1, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, 2, 1)
        self.bn3 = nn.BatchNorm2d(256)

        self.dropout1 = nn.Dropout2d(0.5)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x))) # out_channel: 256
        x = self.dropout1(x)

        x = self.avg_pool(x)  # shape (batch, 256, 1, 1)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        logit = self.fc2(x)  # shape (batch, 10)

        return logit

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad() # 梯度清空
        output = model(data) # 前向传播
        loss = criterion(output, target) # 计算损失
        loss.backward() # 反向梯度计算
        optimizer.step() # 更新参数
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss(reduction='sum')  # 累加总 loss

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # 累加整个测试集的损失
            pred = output.argmax(dim=1, keepdim=True)  # shape (batch, 1), 返回最大置信度的索引
            correct += pred.eq(target.view_as(pred)).sum().item() # view_as(),保持和pred的shape一致

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    # Training settings
    # step1: 命令行参数解析
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=3407, metavar='S',
                        help='random seed (default: 3407)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    # step2: device设置
    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}  # 当使用cuda的时候，加速数据读取的参数
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # step3: 数据准备
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将三通道的像素值 0-255 整数值转为浮点数，并除以255，得到0-1的像素值
        transforms.Normalize((0.1307,), (0.3081,)) # 数据标准化为 正态分布，加速稳定收敛。如果是其它数据集，mean和std还需要重新计算
    ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                              transform=transform)  # 继承了torch.util.data.Dataset类。如果自定义数据集，则只需重写__getitem__()和__len__()方法。
    dataset2 = datasets.MNIST('../data', train=False,
                              transform=transform)
    train_loader = DataLoader(dataset1, **train_kwargs) # 通过 dataset 和 batch_size 参数初始化 dataloader，批次返回数据
    test_loader = DataLoader(dataset2, **test_kwargs)

    # step4: 模型搭建
    model = Net().to(device)

    # step5: 优化器及学习率配置
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma) # decay learning rate by a factor of gamma every step_size epochs

    for epoch in range(1, args.epochs+1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    # 模型导出为 onnx 格式文件。可以 跨平台、多框架 兼容，适用于不同的业界场景。
    x = torch.rand(1, 1, 28, 28).to(device)
    torch.onnx.export(model, x, "minist.onnx")

    # 保存模型 .pt 格式，方便后续在此基础上继续训练
    if args.save_model:
        torch.save(model, "mnist_cnn.pt")


if __name__ == '__main__':
    main()

