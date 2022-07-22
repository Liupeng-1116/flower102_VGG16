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

# data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
# print(os.getcwd())  # D:\Python_tool\Test_VGG  获得当前运行py文件所在的目录
# print(os.path.join(os.getcwd(), "../.."))  # D:\Python_tool\Test_VGG\../..
# print(os.path.abspath(os.path.join(os.getcwd(), "../..")))


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 定义设备
    print(f"now using  {device}  device.")
    
    # 训练、测试集预处理
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),  # 随机按给定尺寸裁剪PIL图像
                                     transforms.RandomHorizontalFlip(),  # 默认以0.5概率随机水平翻转PIL图像
                                     transforms.ToTensor(),  # 转化tensor
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),  # 给定每个通道均值标准差进行标准化处理
        "val": transforms.Compose([transforms.Resize((224, 224)),  # 只按给定大小调整图像
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

    # 获取数据集的前期准备（路径）
    data_root = os.path.abspath(os.path.join(os.getcwd()))  # 获取到当前脚本运行目录
    # D:\Python_tool\Test_VGG
    image_path = os.path.join(data_root, "data_set", "flower_data")  # 组装得到数据存储路径
    # D:\Python_tool\Test_VGG\data_set\flower_data
    assert os.path.exists(image_path), f"{image_path} path does not exist."
    # 路径文件不存在，抛出异常

    # 一、获取训练数据
    # 1、读取图像，转换为tensor
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    # datasets.ImageFolder是通用的数据加载器，取已经按类别放好图像的数据。
    # 与DataLoader不一样。有点类似：
    # train_file = datasets.MNIST(root='./dataset/', train=True,transform=transforms.ToTensor(),download=True)
    # 但这是直接下载数据集的。上面是加载已经自己下载放置好的数据
    train_num = len(train_dataset)  # 训练数据量

    # 2、图像类别索引
    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4} 上面是数字代表的花卉类别
    flower_list = train_dataset.class_to_idx
    # 返回一个类别名(文件夹名）为key,类别索引为val的字典 (class_name, class_idx)
    # 但是，此时是类别名对应索引。
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # 重新改写为（class_idx, class_name)的字典。此时是按索引找类名。
    # 把类名与索引的对应写入json文件
    json_str = json.dumps(cla_dict, indent=4)  # indent表示缩紧程度
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    # 3、生成训练数据batch
    batch_size = 64
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    # 取当前CPU数量、批大小、8之间最小的，似乎是在说明什么多设备并行训练？？？
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size, shuffle=True,
                              num_workers=0)
    # num_workers 表示使用线程个数，window系统无法设置非0值，默认为0（只在主进程加载）

    # 4、加载验证（测试）数据
    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = DataLoader(validate_dataset,
                                 batch_size=batch_size, shuffle=False,
                                 num_workers=0)
    print(f"using {train_num} images for training, {val_num} images for validation.")

    # test_data_iter = iter(validate_loader)
    # test_image, test_label = test_data_iter.next()

    # 5、配置VGG网络参数
    model_name = "vgg16"  # 确定用多大深度的VGG网络
    # 实例化vgg网络的方法，调用model_name表示要使用哪一个VGG配置，分类的个数，是否进行初始化，最后会保存在**kwargs这个可变长度字典中
    net = model.vgg(model_name=model_name, num_classes=102, init_weights=True)
    net.to(device)  # 上GPU训练
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    # 6、设置训练过程数值
    epochs = 30
    best_acc = 0.0  # 一个可覆盖变量，用来后续测出的准确度对其进行覆盖
    save_path = f"./{model_name}Net.pth"  # 保存网络路径
    train_steps = len(train_loader)  # 训练集中一共有多少个batch。也就是一个EPOCH中要经过的步数
    running_loss_list = []
    val_loss_list = []
    running_acc_list = []
    val_acc_list = []

    # 7、开始训练
    for epoch in range(epochs):
        net.train()  # 设置网络处于训练模式
        running_loss = 0.0
        running_loss_epoch = 0.0
        running_correct = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)  # 训练进度条（一次EPOCH进度）
        # train_bar 是一个返回的可迭代对象,根据当前epoch中train_loader的学习显示进度条

        for step, data in enumerate(train_bar):
            images, labels = data  # data是train_loader中的数据，是一个元组，包括图像信息和标签信息。
            # 梯度0
            optimizer.zero_grad()
            images = images.to(device)  # 数据上GPU
            labels = labels.to(device)
            outputs = net(images)  # 输出神经网络预测值
            running_correct += (outputs.argmax(1) == val_labels).sum().item()
            loss_batch = loss_function(outputs, labels)  # 损失函数值
            loss_batch.backward()  # 计算梯度
            running_loss_epoch += loss_batch
            optimizer.step()
            # 训练数据信息显示
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss_batch)
            # desc是进度条的前缀，或者叫进度条名称
        running_loss += running_loss_epoch.item() / train_num  # 一次epoch训练中的损失函数信息
        running_loss_list.append(running_loss)  # 每个epoch的损失值
        running_accurate = running_correct / train_num
        running_acc_list.append(running_accurate)  # 每个epoch对训练数据集的精准度

        # v验证
        net.eval()  # 进入验证模式，调整dropout层作用
        val_loss_epoch = 0.0
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():  # 关闭梯度，不再更新数据
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:   # 不需要step了
                val_images, val_labels = val_data
                val_images = val_images.to(device)
                val_labels = val_labels.to(device)
                outputs = net(val_images)
                val_loss_batch = loss_function(outputs, val_labels)  # 损失函数值
                val_loss_epoch += val_loss_batch
                # predict_y = torch.max(outputs, dim=1)[1]
                # acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                acc += (outputs.argmax(1) == val_labels).sum().item()
                # item() 不等于items()。item()是将只有一个元素的numpy数组或者tensor张量转换为标量的方法

        val_loss_epoch = val_loss_epoch.item() / val_num
        val_loss_list.append(val_loss_epoch)  # 每个epoch对验证数据集的损失函数值
        val_accurate = acc / val_num
        val_acc_list.append(val_accurate)  # 每个epoch对验证数据集的精准度
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate  # 更新当前最有精确度
            torch.save(net.state_dict(), save_path)  # 存储模型参数

    print('Finished Training')
    print('See Chart ！')
    # ------------------- #
    # markers = {'running': 'o', 'val': 's'}
    x = np.arange(len(running_loss_list))  # 横坐标为epoch的值，经过多少个epoch精度是多少
    y1 = running_loss_list
    y2 = running_acc_list
    y3 = val_loss_list
    y4 = val_acc_list
    plt.plot(x, y1, label='running loss', marker='+')
    plt.plot(x, y3, label='val loss', linestyle='r--', marker='x')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.ylim(0, 0.1)
    plt.legend(loc='lower right')
    plt.show()

    plt.plot(x, y2, label='running acc', marker='+')
    plt.plot(x, y4, label='val acc', linestyle='b--', marker='x')
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.ylim(0, 1.0)
    plt.legend(loc='lower right')
    plt.show()


if __name__ == '__main__':
    main()
