# 使用test数据集对模型进行测试
import os
import sys
import json  # json数据交换模块
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm  # 进度条提示模块
import model
from torch import utils
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import numpy as np


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"now using  {device}  device.")

    # 测试集预处理
    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # 获取数据集的前期准备
    data_root = os.path.abspath(os.path.join(os.getcwd()))  # 获取到当前脚本运行目录
    # D:\Python_tool\Test_VGG
    image_path = os.path.join(data_root, "data_set", "flower_data")  # 组装得到数据存储路径
    # D:\Python_tool\Test_VGG\data_set\flower_data
    assert os.path.exists(image_path), f"{image_path} path does not exist."
    # 路径文件不存在，抛出异常

    # 1、读取图像，转换为tensor
    test_dataset = datasets.ImageFolder(root=os.path.join(image_path, "test"),
                                        transform=data_transform)
    test_num = len(test_dataset)  # 测试数据量

    # 2、读取训练时保存的图像类别索引json文件
    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), f"file: '{json_path}' dose not exist."
    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # 3、生成数据batch
    batch_size = 64
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    # 取当前CPU数量、批大小、8之间最小的，似乎是在说明什么多设备并行训练？？？
    print(f"Using {nw} dataloader workers every process")
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=0)

    # 4、使用之前保存的VGG网络参数
    # 因为之前只保存了参数，所以先生成网络实例，再读取参数信息
    net_model = model.vgg(model_name="vgg16", num_classes=102).to(device)
    # 此时已经无须再初始化权重参数
    # 读取参数信息
    weights_path = "./vgg16Net.pth"
    assert os.path.exists(weights_path), "file: '{weights_path}' dose not exist."
    net_model.load_state_dict(torch.load(weights_path, map_location=device))
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net_model.parameters(), lr=0.0001)

    # 5、设置测试过程数值
    best_acc = 0.0  # 一个可覆盖变量，用来后续测出的准确度对其进行覆盖
    test_steps = len(test_loader)  # 训练集中一共有多少个batch。也就是一个EPOCH中要经过的步数

    # 7、验证
    net_model.eval()
    test_acc = 0.0
    test_loss = 0.0
    correct = 0.0
    total = 0.0
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            labels = labels.to(device)
            output = net_model(data)
            test_loss += loss_function(output, labels)
            correct += (output.argmax(1) == labels).sum()  # 预测正确的总数
            total += data.size(0)  # 送入验证的数据总量
    print(f'用于测试的花卉图像数据共有： {total}  个')
    loss = test_loss.item() / len(test_loader)
    print(f'测试集损失函数值： {loss}')
    acc = correct.item() / total
    print('测试集精确度：  {%.3f}  （保留三位小数）' % acc)
# -----------------------------------------------------------#
    print('Finished Test')


if __name__ == '__main__':
    main()
