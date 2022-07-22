import torch.nn as nn
import torch

# official pretrain weights
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
}


# 定义VGG类，继承nn.Module父类
class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=False):
        """
        :param features: make_features(cfg: list)函数生成的提取特征网络结构
        :param num_classes: 进行分类的类别数
        :param init_weights: 是否初始化权重
        """
        super().__init__()  # 父类初始化
        self.features = features
        self.classifier = nn.Sequential(   # 分类器网络结构
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            # inplace 参数只是决定是否对输入数据进行操作后的覆盖（无中间变量）
            # 比如 x = x+1,这就是inplace=True。而y=x+1,x=y,就是非原地操作(中间变量占用内存），就是False.
            nn.Dropout(p=0.5),  # 50%的比例进行随机失活
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)  # num_classes输出节点个数就是分类类别的个数
        )
        if init_weights:  # init_weights为True则进行初始化权重
            self._initialize_weights()

    # 正向传播的过程
    def forward(self, x):  # x表示输入的图像数据
        # N x 3 x 224 x 224
        # 先进行卷积、池化
        x = self.features(x)  # 将数据传入features结构得到输出x
        # N x 512 x 7 x 7
        # 将输出进行展平处理
        x = torch.flatten(x, start_dim=1)  # start_dim=1，从第一维开始展，也就是只保留每次送入数据个数
        # N x 512*7*7
        # 全连接层处理
        x = self.classifier(x)  # 将特征矩阵输出的提前定义好的分类网络结构classifier函数中，得到输出
        return x

    # 权重初始化函数
    def _initialize_weights(self):
        for m in self.modules():  # Module父类内置函数，会返回当前网络中所有模块

            if isinstance(m, nn.Conv2d):  # 遍历到卷积层
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)  # Xavier(均匀分布）初始化卷积核权重
                if m.bias is not None:  # 如果卷积核使用了偏置
                    nn.init.constant_(m.bias, 0)  # 偏置默认使用常数初始化0

            elif isinstance(m, nn.Linear):  # 遍历到全连接层
                nn.init.xavier_uniform_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.01)  正态分布（均值0，标准差为0.01）
                nn.init.constant_(m.bias, 0)  # 则把偏置默认初始化为0


# 提取生成特征网络
def make_features(cfg: list):  # 传入的是配置变量，是列表类型
    layers = []  # 按顺序堆叠层
    in_channels = 3  # 输入RGB图像，通道3

    for v in cfg:  # 遍历配置列表
        if v == "M":  # 如果当前的配置元素是一个M字符的话
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]  # 表示该层是最大池化层，则创建一个最大池化下采样层
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, stride=1, padding=1)  # 创建一个卷积操作，stride默认是1，不需要写进去
            layers += [conv2d, nn.ReLU(True)]   # 每一个卷基层后配备一个RELU层
            in_channels = v   # 卷积输出的通道数，就是下一层输入的通道数
    return nn.Sequential(*layers)
    # nn.Sequential()函数可以接受*args非关键字参数

cfgs = {
    # vgg11表示11层的网络，vgg16就是16层。。。。
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    # 数字代表的是卷积层卷积核的个数，M表示是的是池化层的一个架构。
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


# 定义vgg()函数，实例化给定的配置模型
def vgg(model_name="vgg16", **kwargs):  # model_name表示需要实例化哪个配置
    assert model_name in cfgs, "Warning: model number {} not in cfgs dict!".format(model_name)

    cfg = cfgs[model_name]  # 根据需要的模型名称，选定对应网络结构参数

    model = VGG(make_features(cfg), **kwargs)
    # VGG类实例化。
    # 设置**kwargs，接受设置num_classes=1000, init_weights=False参数的可变长度字典变量
    return model