---
date: 2025-04-02 11:09:52
title: 图像分类
categories: [CV, 图像分类]
tag: AI_Module
---

# 图像分类_overview

## 图像分类

分类模型给图像分配多个标签，每个标签的概率值不同，如dog:1%，cat:4%，panda:95%，根据概率值的大小将该图片分类为panda，那就完成了图像分类的任务。

## 常用数据集

### CIFAR-10和CIFAR-100数据集解释

CIFAR-100 = Canadian Institute For Advanced Research - 100 classes 

CIFAR：加拿大高等研究院

100： 代表这个数据集中包含的 **100个细粒度的类别**（classes）。这与它的前身CIFAR-10（包含10个类别）形成了直接对比。

### 细粒度解释

“细粒度”是相对于“粗粒度”而言的，它描述的是一种**更精细、更具体、更关注细微差别**的分类或分析级别。

您可以把它想象成观察事物的“放大镜倍数”：

- **粗粒度**：低倍数放大镜，看大致的轮廓和类别。
  - **例如**：识别一辆“车”、一只“鸟”、一条“狗”。
- **细粒度**：高倍数放大镜，看具体的型号、品种或子类型。
  - **例如**：识别这辆车是“2012款奥迪A6”还是“2020款特斯拉Model 3”；这只鸟是“北美红雀”还是“美洲知更鸟”；这条狗是“哈士奇”还是“阿拉斯加雪橇犬”。

### torchvision 加载数据集的工具包

```
import torchvision
"""
 使用CIFAR10这个数据集,
 root="./dataset": 会在当前目录下创建dataset文件夹，同时把数据保存进去
 train=True:       这是一个训练集，为False, 则表明这是一个测试集
 download=True:    数据集会从网上下载
"""
train_set = torchvision.datasets.CIFAR10(root="./dataset",
                                              train=True,
                                              download=True)
test_set = torchvision.datasets.CIFAR10(root="./dataset",
                                             train=False,
                                             download=True)
```

### ImageNet解释

ImageNet数据集是ILSVRC竞赛使用的是数据集，包含了超过1400万张全尺寸的有标记图片

ILSVRC全称ImageNet Large-Scale Visual Recognition Challenge大规模视觉识别挑战赛

### **当年获胜团队所提出的神经网络模型的名字**

（NEC-UiUI, XRCE, AlexNet, ZFNet, VGG, GoogleNet, ResNet, SENet）

| 年份 | 模型名称      | 名称由来类型      | 含义                                            |
| ---- | ------------- | ----------------- | ----------------------------------------------- |
| 2010 | **NEC-UiUI**  | 机构名称          | NEC公司智能图像理解实验室                       |
| 2011 | **XRCE**      | 机构名称          | 欧洲施乐研究中心                                |
| 2012 | **AlexNet**   | 核心作者名        | 主要作者Alex Krizhevsky                         |
| 2013 | **ZFNet**     | 作者姓氏首字母    | 作者Zeiler和Fergus                              |
| 2014 | **VGG**       | 研发小组名        | 牛津大学视觉几何组（Visual Geometry Group）     |
| 2014 | **GoogLeNet** | 公司名 + 致敬前辈 | Google + LeNet，其核心结构叫Inception Module    |
| 2015 | **ResNet**    | 核心技术概念      | 残差网络（Residual Network）                    |
| 2017 | **SENet**     | 核心技术概念      | 挤压-激励网络（Squeeze-and-Excitation Network） |

1. NEC-UiUI (2010) & XRCE (2011)
    背景：这两年（2010, 2011）的获胜者还没有使用深度学习，而是采用传统的计算机视觉方法（如稀疏编码+SVM）。

名字由来：这两个名字直接来自于研究机构的名称。

NEC-UiUI: NEC 是日本电气株式会社，UiUI 是其位于美国新泽西的 Laboratory of Intelligent Image Understanding 的缩写。

XRCE: 是 Xerox Research Centre Europe（欧洲施乐研究中心）的缩写。

2. AlexNet (2012) - 深度学习革命的起点
    名字由来：以其第一作者 Alex Krizhevsky 的名字命名。他是这篇开创性论文的主要贡献者。他的导师Geoffrey Hinton和另一位学生Ilya Sutskever也是合著者。
3. ZFNet (2013)
    名字由来：以其两位作者 Zeiler 和 Fergus 的姓氏首字母命名。Matthew Zeiler 和 Rob Fergus 来自纽约大学。这个模型是在AlexNet基础上的改进，并通过“反卷积网络”提供了对CNN工作原理的重要可视化见解。
4. VGG (2014)
    名字由来：来源于其研发机构——英国牛津大学的视觉几何组（Visual Geometry Group）。这个模型以其极深的、仅由3x3卷积层堆叠而成的简单统一结构而闻名（如VGG-16, VGG-19）。

5. GoogLeNet (2014) / Inception
    名字由来：这是一个双关语。

Google：表明其研发团队是Google（现为Google Research）。

LeNet：向Yann LeCun早年的开创性卷积网络LeNet-5致敬。

它更核心的技术名称是 Inception，得名于模型中的核心模块“Inception Module”，这个模块的名字来源于互联网上著名的“We need to go deeper”的梗图（源自电影《盗梦空间》，英文名Inception）。

6. ResNet (2015)
     名字由来：源于其最核心的创新——“残差模块”（Residual Network）。由何恺明等人来自微软亚洲研究院（MSRA）的团队提出。它通过“快捷连接”（Shortcut Connection）解决了极深网络难以训练的梯度消失问题，使得网络可以深达上百甚至上千层。
7. SENet (2017)
     名字由来：代表“Squeeze-and-Excitation Network”（挤压-激励网络）。由Momenta公司（一家中国自动驾驶公司）的研究团队提出。它的创新在于在传统卷积模块中加入了“SE模块”，该模块可以自适应地校准通道特征响应，从而显著提升模型的性能。SENet通常不是作为一个独立的网络，而是作为一个即插即用的模块，可以嵌入到ResNet等现有网络中（例如SE-ResNet）。



### 鲜花分类数据集

鲜花分类数据集是包含 5 种类型的图像数据集，主要用于图像分类，其共有3670 张图像，其中训练图像和测试图像分别为 3306 张和 364张，主要分为五类鲜花，分别为菊花，蒲公英，玫瑰，向日葵和郁金香。

使用以下API将这些数据解析出来：

```
ImageFolder(root, transform=None)
参数的意义如下所示：

root：在root指定的路径下寻找图片

transform：对图像数据进行处理，比如类型转换，尺寸的调整等
```

具体实现如下所示：

```
from torchvision.datasets import ImageFolder
# 指定数据集路径
flowers_train_path = './dataset/flower_datas/train/'
flowers_test_path = './dataset/flower_datas/val/'
# 先将数据转换为tensor类型，并调整数据的大小为224x224
dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),      
    torchvision.transforms.Resize((224,224))
])
# 获取训练集数据和测试集数据
flowers_train = ImageFolder(flowers_train_path, transform=dataset_transform)
flowers_test = ImageFolder(flowers_test_path,transform=dataset_transform)
```

可以获取训练集和测试集样本数量并进行展示：

```
len(flowers_train.imgs) # 3306
len(flowers_test.imgs) # 364
# 图像展示：随机指定某一幅图片进行可视化
import matplotlib.pyplot as plt
plt.imshow(flowers_train.__getitem__(3000)[0].permute(1,2,0))
```

总结：

图像分类的定义

从给定的类别集合中为图像分配对应的类别标签

常用数据集：cifar数据集，imageNet（大规模视觉识别竞赛专用），鲜花集

数据集加载的工具：torchvision，ImageFolder

历史冠军网络模型：AlexNet、VGG、googleNet、ResNet、SEnet

# 图像分类_AlexNet



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

### 数据读取

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

#### 220x220x3（彩色图）：

相当于宽高220x220个像素点，每个像素点有三个通道（三原色的亮度，明暗程度）RGB(red的强度，green的强度，blue的强度)；示例：（255,0,0），只红色发光，绿和蓝不发光。（255,255,0）蓝色不发光，红色+绿色=黄色光。

#### 220x220x1（灰度图-灰色的亮度）：

#### 每一个像素点，都只有一个元素，就是灰色的亮度（0-255）。0是纯黑，255是纯白，中间是不同深浅的灰色。

#### 220x220x1（二值图-黑或者白的亮度）：

#### 220x220x1（索引图-索引指向三原色）：

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

![图片的替代文本](https://bob-blog-image.oss-cn-shanghai.aliyuncs.com/bird.jpeg)

