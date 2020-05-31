---
description: >-
  网络上有很多Keras的入门指南，所以我将跳过如何安装软件等基础内容，只讨论具体案例。如果读者不知道如何安装Keras，可以访问Keras官方网站的快速入门，他们有提供中文指南，上面有详细的安装指导。
---

# 附录2：Keras入门指南

Keras严格来说不算机器学习软件，也不是机器学习框架，因为它本身并不提供机器学习库，它应该算是一个对用户有好的规范API调用接口。它和机器学习库的关系有点像linux的图形界面和后台shell命令。常见的机器学习库有TensorFlow, CNTK, 或者Theano。它们之间是相互独立的机器学习工具。只要读者愿意，抛开Keras，直接使用它们是完全可行的，但是有一个问题，它们的参数太多，应用时规则过于复杂。

对大多数人而言，需要解决的问题只是使用神经网络来自学习某个具体问题模型的函数，至于其中到底要使用哪种优化算法来使得学习效率更高更快他们并不关心。而往往大部分问题并不是太复杂，已知的一些简单算法就足以应付。在这种情况下，直接去使用机器学习库，把简单问题当复杂问题一样来编程操作就显得多余。而且现在常见的机器学习库不下十余种，虽然它们解决的问题相类似，但是每个软件都有自己的使用规范，如果让用户逐个都学习一遍，那就太不人道了。Keras就是为了解决这个问题而诞生的，它提供了更高层面的API，将机器学习库TensorFlow, CNTK, 和Theano作为后端。用户只要调用统一的Keras接口，就可以随意操作TensorFlow, CNTK, 或者Theano来进行机器学习。这些库之间规范上的差异由Keras来统一解决，这大大方面了普通用户。用户获得的一个最大好处就是可以把自己的想法（idea）迅速转换为应用模型，用最短的时间来验证自己的想法。

## 安装

首先要安装Keras的后端学习库，Keras本身只是这些机器学习库的顶层API。推荐使用TensorFlow，当然使用其它的库并不会给机器学习的结果带来影响。使用[pip](https://pip.pypa.io/en/stable/)工具来安装Python是非常方便的，为了使得安装更稳定，这里额外使用Python的镜像源来安装TensorFlow。国内能稳定访问的镜像源如下（只列出支持https协议的镜像源）：

> 清华大学：[https://pypi.tuna.tsinghua.edu.cn/simple](https://pypi.tuna.tsinghua.edu.cn/simple)
>
> 阿里云：[https://mirrors.aliyun.com/pypi/simple/](https://mirrors.aliyun.com/pypi/simple/)
>
> 加利福尼亚大学：[https://www.lfd.uci.edu/~gohlke/pythonlibs/](https://www.lfd.uci.edu/~gohlke/pythonlibs/)
>
> 中国科技大学 [https://pypi.mirrors.ustc.edu.cn/simple/](https://pypi.mirrors.ustc.edu.cn/simple/)
>
> 豆瓣：[https://pypi.douban.com/simple/](https://pypi.douban.com/simple/)

安装TensorFlow：

```text
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tensorflow
```

需要小心，最新版本的TensorFlow不支持32位的Python，如果读者的Python是32位的话，会遇到异常，需要重新安装64位的Python版本。如果用户拥有NVIDIA的显卡，可以安装TensoFlow-GPU版，这个版本可以使用显卡来加速运算。需要注意，不是所有的显卡都支持GPU加速（需要显卡支持CUDA），请安装前先确认自己的显卡是否在[支持列表](https://developer.nvidia.com/cuda-gpus)中。TensoFlow-GPU的安装更多的是涉及显卡的驱动安装与配置，与本书关系不大，不再花费篇幅累述，读者如有兴趣可以参考TensorFlow中文社区的[安装指南](http://www.tensorfly.cn/tfdoc/get_started/os_setup.html)。

安装Keras：

```text
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple keras
```

## 验证

打开python环境，输入：

```text
import keras
```

显示`Using TensorFlow backend.`表示安装成功。

## 基本操作：

_**案例1**_ **\*\*\_**使用前馈网络实现线性回归（函数拟合）\*\*\_

_**案例2**_ **\*\*\_**使用前馈网络实现数据分类\*\*\_

_**案例3 使用卷积网络实现图像分类**_

