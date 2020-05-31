# temp

机器学习能够在近些年获得如此迅猛的发展，和人们能更便捷地获取大量的数据是密不可分的。我们接下来要实现的基础智能AI也同样需要大量的数据。作为AI的学习教材，可以从[https://www.u-go.net/gamerecords/](https://www.u-go.net/gamerecords/)获取历年来在KGS上7段以上选手之间的对弈棋谱。网站上有“.zip”、“.tar.gz”和“.tar.bz2”三种格式。为了方便，我们在windows上处理这些数据，因此，我们只使用“.zip”格式的文件。如果你愿意，完全可以手工逐个点击下载，不过为了方便和快速，这里提供一个Python小程序，方面地获取所有的“.zip”格式链接。 右键浏览器，把[https://www.u-go.net/gamerecords/的网页文件保存在myGO\SGF\_Parser文件夹下，使用默认文件名“u-go.net.html”保存。](https://www.u-go.net/gamerecords/的网页文件保存在myGO\SGF_Parser文件夹下，使用默认文件名“u-go.net.html”保存。) 编辑Python文件fetchLinks.py：

在cmd窗口里，执行python fetchLinks.py &gt; zip.link 打开“zip.link”文件，将全部内容复制后粘贴到迅雷中下载，文件请保存在“myGO\SGF\_Parser\sgf\_data\”。 全选所有下载下的zip文件，右键，选择7-zip进行解压，选择“提取到当前目录”，这样，在“myGO\SGF\_Parser\sgf\_data\”目录下就会有全部待解析处理的sgf文件了。

需要注意一点，调用import keras时，如果本地没有数据时，数据不会下载，只有在实际赋值使用时才会

该选择结构可以灵活对theano和tensorflow两种backend生成对应格式的训练数据格式。举例说明：'th'模式，即Theano模式会把100张RGB三通道的16×32（高为16宽为32）彩色图表示为下面这种形式（100,3,16,32），Caffe采取的也是这种方式。第0个维度是样本维，代表样本的数目，第1个维度是通道维，代表颜色通道数。后面两个就是高和宽了。而TensorFlow，即'tf'模式的表达形式是（100,16,32,3），即把通道维放在了最后。这两个表达方法本质上没有什么区别。

策略梯度 就是根据结果对已经下过的棋进行取舍 q-learning 就是学习输入棋局，合法步预测出输赢价值 上面两种的本质都差不多，就是根据结果对监督学习的学习目标进行归类（好的与坏的），然后再学习

