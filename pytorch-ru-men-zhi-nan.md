---
description: >-
  读者可以直接登陆官网参考Pytorch的官方入门指南（https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html），本文可以看作是一个精简后的官方指南中文版。
---

# Pytorch入门指南

对于经费不足的个人或者学术机构而言，Pytorch是有其优势的。TensorFlow总是会集成许多CPU和GPU的最新技术，所以一些老旧的CPU和GPU因为不支持这些新的技术特性，可能会无法运行TensorFlow，而Pytorch则几乎是来者不拒的。Pytorch比TensorFlow要轻量许多，这一点总是受到个人和非商业部门欢迎的，在有限的资源上，人们会倾向于更关心自己的想法能否快速实现，而不是把时间浪费在老旧机器的卡顿上。下表是将Pytorch和TensorFlow做了一个简单对比，供读者参考。

|  | Pytorch | TensorFlow |
| :--- | :--- | :--- |
| **主要面向** | 研究 | 工业界，商用 |
| **部署** | Torch Serve | TensorFlow Serve |
| **适用场景** | 个人开发或者科研 | 企业 |
| **可视化** | Visdom | Tensorboard |

本书的编程部分是倾向于偏袒编程初学者的，因此几乎所有的程序都是以windows平台为基础，如果读者是资深的程序员，将代码从windows迁移至其它平台应该不会存在困难。Pytorch的windows版本仅支持Python 3.x，读者在使用Pytorch前需要确定自己使用的是Python 3.x，很多情况是运行的服务器上同时安装了Python 3.x和Python 2.x。 

在开始介绍Pytorch使用方法前，读者需要知道GPU+CUDA的组合是目前为机器学习加速计算的最常见组合。如果你的计算平台装有一块显卡，并且支持Nvidia的CUDA，就可以在代码里加上下面这句话，使得神经网络在训练时能够利用CUDA特性使得训练速度有显著的提升。

```text
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))
```

我们以一个3层神经网络网络为例，输入层是6个神经元（输入层∈$$R^6$$），隐藏层是8个神经元（隐藏层∈$$R^8$$），输出层是5个神经元（输出层∈$$R^5$$）。每个神经元采用Relu激活函数。

![&#x7F16;&#x7A0B;&#x793A;&#x4F8B;](.gitbook/assets/pic1.png)

在一切开始以前，按Python编程的惯例，要引入一些必要的库。

```text
import torch
from torch import nn    #1
```

1. **nn里包含了所有的构成神经网络基础类，基础类都可以通过采用“nn.模块名”的方式来调用。**

接着我们先定义一个神经网络的类结构作为神经网络的模型抽象，方便后续调用。

```text
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()    #1
        self.flatten = nn.Flatten()    #2
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(6, 8),    #3
            nn.ReLU(),    #3
            nn.Linear(8, 5),    #3
            nn.ReLU()    #3
        )
```

1. 使用NeuralNetwork的父类nn.Module来初始化我们自己的神经网络应用类；
2. 通常神经网络的输入可能是一张图片或者更高维度的数据，我们可以使用flatten方法把输入数据转成一维的数据。如果你的输入数据已经提前做了转换，这一步可以省略；
3. 层与层之间采用全连接的方法，并使用Relu作为神经元的激活函数。

有了网络，我们再为网络定义前向计算的方法。

```text
class NeuralNetwork(nn.Module):
    def __init__(self):
        ...

    def forward(self, x):
        x = self.flatten(x)    #1
        logits = self.linear_relu_stack(x)    #2
        return logits    #2
```

1. 前向运算前，先将数据打平，如果数据已经进行过了预处理，这一步便不会产生什么效果；
2. 采用之前定义好的网络结构进行计算，并对外输出结果。

有了抽象的模型，接着便是实例化这个抽象模型。

```text
model = NeuralNetwork()
```

接着就是为模型定义一下损失函数和采用什么优化算法来做反向梯度计算。

```text
loss_fn = nn.CrossEntropyLoss()    #1
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)    #2
```

1. 在例子中









如果读者已经熟悉了Keras的用法，会发现Pytorch和Keras在使用上是十分相似的，所以在了解了基本用法后，在工具之间过渡将十分方便。

