# auther  : qian_yu wang
# content :
# data    : 2023/4/2
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, inchannel, outchannel, res=True, stride=1):
        super(Block, self).__init__()
        self.res = res  # 是否带残差连接
        self.left = nn.Sequential(
            # 之所以不用定义卷积核是因为卷积核的值在这里是随机的
            nn.Conv2d(inchannel, outchannel, kernel_size=3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(outchannel),
            # inplace = True ,会改变输入数据的值,节省反复申请与释放内存的空间与时间,只是将原来的地址传递,效率更好
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(outchannel),
        )
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, bias=False),
                nn.BatchNorm2d(outchannel),
            )
        else:
            self.shortcut = nn.Sequential()

        self.relu = nn.Sequential(
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.left(x)
        if self.res:
            out += self.shortcut(x)
        out = self.relu(out)
        return out


class MyModel(nn.Module):
    # 修改
    def __init__(self, cfg=None, res=True, in_channel=3):
        super(MyModel, self).__init__()
        if cfg is None:
            cfg = [64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M']
        self.res = res  # 是否带残差连接
        self.cfg = cfg  # 配置列表
        self.inchannel = in_channel  # 初始输入通道数
        self.futures = self.make_layer()
        # 构建卷积层之后的全连接层以及分类器：
        self.classifier = nn.Sequential(nn.Dropout(0.4),  # 两层fc效果还差一些
                                        nn.Linear(4 * 512, 10), )  # fc，最终Cifar10输出是10类

    def make_layer(self):
        layers = []
        for v in self.cfg:
            # 池化层
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            # 卷积层
            else:
                layers.append(Block(self.inchannel, v, self.res))
                self.inchannel = v  # 输入通道数改为上一层的输出通道数
        # 如果*号加在实参上，代表的是将输入迭代器拆成一个个元素。
        # 不加*号，会报错 TypeError: list is not a Module subclass
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.futures(x)
        # view(out.size(0), -1): change tensor size from (N ,H , W) to (N, H*W)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
