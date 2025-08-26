---
date: 2025-03-19 05:44:06
title: 图像分类_02_AlexNet
categories: [CV, 图像分类, AlexNet]
tag: AI_Module
---

## AlexNet网络结构

2012年，AlexNet横空出世，该模型的名字源于论文第一作者的姓名Alex Krizhevsky 。AlexNet使用了8层卷积神经网络，以很大的优势赢得了ImageNet 2012图像识别挑战赛。它首次证明了学习到的特征可以超越手工设计的特征，从而一举打破计算机视觉研究的方向。

在pytorch中构建AlexNet模型，继承自类nn.Module，实现init和forward方法即可，具体如下所示：

```
from torch import nn

# 创建AlexNet网络结构
class Alexnet(nn.Module):
    # 初始化过程，网络层的定义,指明输入数据的通道数和输出的类别个数
    def __init__(self, in_dim, n_class):
        super().__init__()
        # 卷积部分
        self.conv = nn.Sequential(
            # 卷积层：96个卷积核，卷积核为11*11，步幅为4：(3,227,227) -> (96,55,55)
            nn.Conv2d(in_channels=in_dim,
                      out_channels=96,
                      kernel_size=11,
                      stride=4,
                      padding=0),
            # 激活函数relu
            nn.ReLU(True),
            # 池化:窗口大小为3*3、步幅为2：(96,55,55)->(96,27,27)           
            nn.MaxPool2d(3, 2),
            # 卷积层：256个卷积核，卷积核为5*5，步幅为1，padding为2：(96,27,27) -> (256,27,27)
            nn.Conv2d(96, 256, 5, stride=1, padding=2),
            # 激活函数relu
            nn.ReLU(True),
            # 池化:窗口大小为3*3、步幅为2：(256,27,27) -> (256,13,13)
            nn.MaxPool2d(3, 2),
            # 卷积层：384个卷积核，卷积核为3*3，步幅为1，padding为1：(256,13,13) -> (384,13,13)
            nn.Conv2d(256, 384, 3, stride=1, padding=1),
            # 激活函数relu
            nn.ReLU(True),
            # 卷积层：384个卷积核，卷积核为3*3，步幅为1，padding为1：(384,13,13) -> (384,13,13)
            nn.Conv2d(384, 384, 3, stride=1, padding=1),
            # 激活函数是relu
            nn.ReLU(True),
            # 卷积层：256个卷积核，卷积核为3*3，步幅为1，padding为1:(384,13,13) -> (256,13,13)
            nn.Conv2d(384, 256, 3, stride=1, padding=1),
            # 激活函数relu
            nn.ReLU(True),
            # 池化:窗口大小为3*3、步幅为2:(256,13,13) -> (256,6,6)
            nn.MaxPool2d(3, 2))
        # 全连接层部分
        self.fc = nn.Sequential(
            # 全连接层:4096个神经元
            nn.Linear(9216, 4096), 
            # 激活函数relu
            nn.ReLU(True),
            # 随机失活
            nn.Dropout(0.5), 
            # 全连接层:4096个神经元
            nn.Linear(4096, 4096),
            # 激活函数relu
            nn.ReLU(True), 
            # 随机失活
            nn.Dropout(0.5),
            # 全连接层:n_class个神经元
            nn.Linear(4096, n_class))

    def forward(self, x):
        # 输入数据送入卷积部分进行处理
        x = self.conv(x)
        #将特征图抻成一维向量
        x = x.view(x.size(0), -1)  
        # 全连接层获取输出结果
        output = self.fc(x)
        return output
```

我们构造一个高和宽均为227的单通道数据样本来看一下模型的架构：

```
from torchsummary import summary
import torch
# 模型实例化
net = Alexnet(3,1000)
# 通过net.summay()查看网络的形状
summary(model=net,input_size=(3,227,227),batch_size=1,device='cpu')
```

网络架构如下：

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [1, 96, 55, 55]          34,944
              ReLU-2            [1, 96, 55, 55]               0
         MaxPool2d-3            [1, 96, 27, 27]               0
            Conv2d-4           [1, 256, 27, 27]         614,656
              ReLU-5           [1, 256, 27, 27]               0
         MaxPool2d-6           [1, 256, 13, 13]               0
            Conv2d-7           [1, 384, 13, 13]         885,120
              ReLU-8           [1, 384, 13, 13]               0
            Conv2d-9           [1, 384, 13, 13]       1,327,488
             ReLU-10           [1, 384, 13, 13]               0
           Conv2d-11           [1, 256, 13, 13]         884,992
             ReLU-12           [1, 256, 13, 13]               0
        MaxPool2d-13             [1, 256, 6, 6]               0
           Linear-14                  [1, 4096]      37,752,832
             ReLU-15                  [1, 4096]               0
          Dropout-16                  [1, 4096]               0
           Linear-17                  [1, 4096]      16,781,312
             ReLU-18                  [1, 4096]               0
          Dropout-19                  [1, 4096]               0
           Linear-20                  [1, 1000]       4,097,000
================================================================
Total params: 62,378,344
Trainable params: 62,378,344
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.59
Forward/backward pass size (MB): 11.05
Params size (MB): 237.95
Estimated Total Size (MB): 249.59
----------------------------------------------------------------
```

## 2.鲜花种类识别

AlexNet使用ImageNet数据集进行训练，但因为ImageNet数据集较大训练时间较长，我们使用前面的介绍的鲜花分类数据集来演示AlexNet。读取数据的时将图像高和宽扩大到AlexNet使用的图像高和宽227。

###  数据读取

首先获取数据,为了适应AlexNet对输入数据尺寸的要求，将图像大小调整到227x227的大小，并使用DataLoader进行批次数据的读取：

```
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
# 指定批次大小
batch_size = 2
# 指定数据集路径
flower_train_path = './dataset/flower_datas/train/'
flower_test_path = './dataset/flower_datas/val/'
# 先将数据转换为tensor类型，并调整数据的大小为227x227
dataset_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((227, 227))])
# 获取训练集数据和测试集数据
flower_train = ImageFolder(flower_train_path, transform=dataset_transform)
flower_test = ImageFolder(flower_test_path, transform=dataset_transform)
# 获取数据的迭代
train_loader = DataLoader(dataset=flower_train,
                          batch_size=batch_size,
                          shuffle=True)
test_loader = DataLoader(dataset=flower_ttest
                         batch_size=batch_size,
                         shuffle=False)
```

为了更好的理解，我们将batch数据展示出来：

```
import matplotlib.pyplot as plt
# 遍历每个迭代的数据，将其结果展示出来
for b, (imgs, targets) in enumerate(train_loader):
    # 获取第一个batch的图像
    if b == 1:
        # 将其进行展示
        fig, axes = plt.subplots(1, 2)
        # 遍历batch中的每个图像
        for i in range(batch_size):
            # 图像显示出来
            axes[i].imshow(imgs[i].permute(1, 2, 0))
            # 设置图像标题
            axes[i].set_title(targets[i].item())
        plt.show()
    elif b > 0:
        break
        
```

2.2 模型实例化和参数设置[¶](#22)

```
# 模型实例化:输入数据3通道，进行5类的分类处理
model = Alexnet(3, 5)
# 模型训练的参数设置
# 学习率
learning_rate = 1e-3
# 训练轮数
num_epochs = 10
# 优化算法Adam = RMSProp + Momentum
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# 交叉熵损失函数
loss_fn = torch.nn.CrossEntropyLoss()
```

## 问题记录：

### 假设220x220x1的一个图片，相当于一张不同颜色像素点的图片，为什么要3通道堆叠？

在计算机中，按照颜色和灰度的多少可以将图像分为四种基本类型。

#### 	220x220x3（彩色图）：

相当于宽高220x220个像素点，每个像素点有三个通道（三原色的亮度，明暗程度）RGB(red的强度，green的强度，blue的强度)；示例：（255,0,0），只红色发光，绿和蓝不发光。（255,255,0）蓝色不发光，红色+绿色=黄色光。

#### 	220x220x1（灰度图-灰色的亮度）：

#### 每一个像素点，都只有一个元素，就是灰色的亮度（0-255）。0是纯黑，255是纯白，中间是不同深浅的灰色。

#### 	220x220x1（二值图-黑或者白的亮度）：

#### 	220x220x1（索引图-索引指向三原色）：

索引图像的文件结构比较复杂，除了存放图像的二维矩阵外，还包括一个称之为颜色索引矩阵MAP的二维数组。颜色矩阵MAP的大小由存放图像的矩阵元素值域决定，如矩阵元素值域为[0，255]，则MAP矩阵的大小为256Ⅹ3；如果是[0,500],颜色矩阵map就是500行3列。MAP中每一行的三个元素分别指定该行对应颜色的红、绿、蓝单色值，

MAP中每一行对应图像矩阵像素的一个灰度值**，如某一像素的灰度值为64，则该像素就与MAP中的第64行建立了映射关系，该像素在屏幕上的实际颜色由第64行的[RGB]组合决定。

也就是说，图像在屏幕上显示时，每一像素的颜色由存放在矩阵中该像素的灰度值作为索引通过检索颜色索引矩阵MAP得到。

索引图和彩色图的区别：

### 核心区别对比表

| 特性         | 索引图 (Indexed Color)                                       | 真彩色图 (True Color)                                        |
| ------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **核心原理** | **间接表示**。像素值是一个**编号（索引）**，需要去调色板（MAP）里查表才能得到最终颜色。 | **直接表示**。像素值**本身就是颜色**（通常是R, G, B三个通道的值）。 |
| **数据结构** | 两个部分： 1. **图像矩阵**：一个二维数组，值为索引号。 2. **调色板(MAP)**：一个颜色列表，`[n, 3]`的大小，n是颜色总数。 | 一个部分： **图像矩阵**：一个三维数组 `(高度, 宽度, 3)`，最后一个维度直接存储R、G、B分量。 |
| **颜色数量** | **有限**（通常最多256色）。所有颜色都必须预先定义在调色板中。 | **极其丰富**（约1677万色）。理论上可以是任何颜色，无需预定义。 |
| **文件大小** | **通常较小**。因为每个像素只需要存储一个索引值（通常1字节），而不是3个颜色值。 | **较大**。每个像素需要存储3个值（3字节），文件大小通常是索引图的3倍或更多。 |
| **灵活性**   | **低**。图像只能使用调色板里定义的颜色。如果要显示调色板中没有的颜色，会通过**抖动**等技术来模拟，但效果不完美。 | **极高**。可以平滑地表现任何颜色的渐变，如照片、复杂阴影等。 |
| **常见格式** | **GIF**, **PNG-8**                                           | **PNG-24**, **JPEG**, **BMP**, **TIFF**                      |
| **典型用途** | 标志(Logo)、图标(Icon)、卡通画、屏幕截图、简单图形等**颜色数量少、有大块纯色区域**的图像。 | 摄影作品、艺术创作、                                         |

stride：

stride=2；相当于行（左右）每次移动两格，移动完之后；列（上下）也每次移动两格，移动完。

### 输出图像的大小

N = [  (W-F+2P)/S  ]+1   =图像

1. 输入图像大小: W x W
2. 卷积核大小: F x F
3. Stride: S
4. Padding: P
5. 输出图像大小: N x N

e.g. [ (7 - 3 + 2x0)/2 ] + 1 =3;   输出图像大小：3x3

e.g. 227x227x3；11x11；stride=4；=55x55x96

[(227 - 11 )/4 ]+1 = ；输出图像大小：55x55

