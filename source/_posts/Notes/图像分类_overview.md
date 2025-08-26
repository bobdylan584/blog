---
date: 2025-03-02 11:09:52
title: 图像分类
categories: [CV, 图像分类]
tag: AI_Module
---

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

