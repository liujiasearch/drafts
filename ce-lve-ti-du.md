---
description: >-
  从这一章开始我们将开始接触机器学习的一个分支：强化学习。强化学习主要分成三个大类，策略梯度、DQN和AC算法。这一章先从策略梯度开始介绍，后面两章再分别对DQN和AC算法进行介绍。
---

# 第七章 策略梯度

前面的章节我们已经学习了如何利用监督学习使得神经网络能够下围棋。但是仅仅利用监督学习，我们的智能AI下出的着法还是相当幼稚的。比如：由于人类选手几乎不会浪费棋子在吃掉对方已经死掉的棋子上，所以我们的神经网络基本上学习不到吃子的概念。利用这个优势，人类选手可以很轻易地战胜通过监督学习训练出的AI智能。目前来看，在监督学习的理论没有突破性发展的情况下，现有的方法不是很行得通。我们仅仅能够让AI学到下围棋的“感觉”。

我非常喜欢“理论和实践的辩证统一”这一提法，具体来说就是利用理论来指导实践，再用实践的结果来修正理论。我们熟悉的‘摸着石头过河’这句话其实就是这个思想的一种具体体现， 其本质就是在实践中不断总结经验的一种形象性说法。古人王阳明也把这种思想称作“知行合一”。策略梯度就是利用理论与实践相互作用的关系，来不断提升机器智能的一种方法。

这里我们设计一个小游戏，目的来演示理论与实践的相互作用是如何提升机器智能水平的。游戏双方A和B各执五张牌，从1到5。每回合双方各从五张牌中抽出一张来比大小，点数大的获胜。当然，这个游戏并不具备实际意义，正常人可以很轻易地得到结论：只要每个回合都出5就可以立于不败之地。用这个游戏规则仅仅是为了说明上的方便。假使A和B都不够聪明，他们并不能一目了然地知道应该如何出牌，他们只会以相等的概率随机出牌，但是A和B都会学习，他们能够根据游戏的结果来更新自己的出牌策略。与前面提到的方法进行比对，可以看出，玩家以相等的概率随机挑选一张牌是他们的初始理论，A和B在这个理论指导下进行出牌，出牌是实践这个理论的过程，而游戏胜负的结果就是实践的结果。A和B根据实践结果来更新自己的出牌策略就是用实践来反作用于理论，使得理论逐步趋于完善。

回到游戏本身，以最极端的情况为例，虽然出1到5的概率相同，但是很不巧，A每回合都出的是1，显然B只要出的牌不是1就能获胜。假使A和B玩了10局，A和B分别计算了一下10局内各种数字的出牌次数。

> A以等概率的方式对1到5这5张牌进行选择，A=\[0.2, 0.2, 0.2, 0.2, 0.2\]
>
> B以等概率的方式对1到5这5张牌进行选择，B=\[0.2, 0.2, 0.2, 0.2, 0.2\]

| 回合 | 玩家 | 出牌 | 胜负平 | 玩家 | 出牌 | 胜负平 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | A | 1 | 负 | B | 2 | 胜 |
| 2 | A | 1 | 负 | B | 4 | 胜 |
| 3 | A | 1 | 负 | B | 5 | 胜 |
| 4 | A | 1 | 平 | B | 1 | 平 |
| 5 | A | 1 | 平 | B | 1 | 平 |
| 6 | A | 1 | 负 | B | 2 | 胜 |
| 7 | A | 1 | 负 | B | 3 | 胜 |
| 8 | A | 1 | 负 | B | 3 | 胜 |
| 9 | A | 1 | 负 | B | 5 | 胜 |
| 10 | A | 1 | 负 | B | 2 | 胜 |

A发现，自己出了那么多1，最好的结果也是平局，所以出1看起来应该不是什么好的事情，也许出其它牌可能更好一些。所以A打算调整一下自己的策略，原来出1到5的概率都相同，现在需要调整一下各张牌的出牌概率。首先就是要降低出1的概率。而对于B来说，他发现只要自己不出1都可以获胜，B认为只要少出1就能提高胜率，所以B也调整了自己的出牌策略。

> A调低了出1的概率，相对的，由于概率的总和是1，其它牌出牌的概率就提高了，A=\[0.16, 0.21, 0.21, 0.21, 0.21\]
>
> B也调低了出1的概率，但是B出1并没有什么不利的情况发生，B对出1的概率调整不如A那么大，B=\[0.18, 0.205, 0.205, 0.205, 0.205\]

调整后A和B又玩了10局：

| 回合 | 玩家 | 出牌 | 胜负平 | 玩家 | 出牌 | 胜负平 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | A | 2 | 负 | B | 4 | 胜 |
| 2 | A | 4 | 负 | B | 5 | 胜 |
| 3 | A | 1 | 负 | B | 2 | 胜 |
| 4 | A | 5 | 胜 | B | 3 | 负 |
| 5 | A | 3 | 胜 | B | 1 | 负 |
| 6 | A | 4 | 胜 | B | 3 | 负 |
| 7 | A | 2 | 负 | B | 3 | 胜 |
| 8 | A | 3 | 平 | B | 3 | 平 |
| 9 | A | 5 | 胜 | B | 2 | 负 |
| 10 | A | 3 | 负 | B | 5 | 胜 |

A根据新的游戏结果又统计了一下自己的出牌情况。他发现出1会输，出2和4也输了，出3有胜有负有平局，出5赢了一局，于是A打算根据这个情况继续降低出1的概率，同时也相应地降低出2和4的概率，由于出3有胜有负，不知道什么情况，暂时不做调整，而出了一次5就赢了，看来出5可能是个好事情，自己应该在策略中提高一点出5的概率。对于B来说，他发现自己这一次出1输了，出2有胜有负，出3的时候虽然有胜负，但是胜少负多，出4和出5都赢了，根据这个情况，B继续调低出1的概率，由于对出2的结果不太确定就暂时保持不动，出3虽然互有胜负，但是总体来说是输多赢少，所以也调低了出3的概率，出4和5都赢了，就顺势提高一点出4和5的概率。

> A根据游戏的反馈调整了对1到5这5张牌出牌的概率，A=\[0.14, 0.195, 0.21, 0.195, 0.26\]
>
> B一样根据游戏情况对1到5这5张牌的出牌概率进行了调整，B=\[0.15, 0.205, 0.19, 0.2275, 0.2275\]

A和B更新完自己的策略后，又继续玩这个游戏。显然，除了出5，出别的牌都有可能输给对方，所以长远来看，双方都会不停的提高自己出5的概率，同时降低出其它牌的概率。随着游戏玩的局数越来越多，最终就会发现：偶尔出5偶尔爽，一直出5一直爽。假使A比B先知道了出5就会一直赢，那么B只有出5才能保持不输，通过上述的学习方法，B会一直降低出其它牌的几率，只需要不多的几个回合B就能得到和A一样的策略。这种通过不断实践与反馈的做法就叫做策略梯度。

程序`/myGO/p_g/easy_policy_game.py`对这个游戏进行了模拟，读者有兴趣的话可以自己执行一下代码看看效果。非常有趣的一个现象是，当我们仅更新一方的策略时，策略收敛的非常快，但是缺点是当训练回合不足时有一定概率会收敛到局部最优点。如果我们同时更新双方的策略，虽然策略不会收敛到局部最小，但是收敛的速度相交前一种方法会慢一些。这些内容读者也可以自行实验。我个人推荐结合这两种方案，既同时更新双方的策略，但是也保持一定的比例将最新策略与随机策略进行比较。具体到这个小游戏来说，就是10回合中可以有5回合是A和B进行游戏，还有5回合让A或者B和采用随机策略的C进行游戏。游戏结果仅用于更新A和B的策略，C的策略保持不变。

类比上面这个游戏，我们尝试将这种方式用于神经网络的更新学习，让神经网络在围棋对弈的过程中逐步学会应该如何下棋。从流程上来说，出牌游戏和围棋游戏的学习过程是一样的，都是利用两个自带初始策略的游戏双方进行多局游戏，然后根据游戏的结果逐步更新原有策略。

![&#x51FA;&#x724C;&#x6E38;&#x620F;&#x4E0E;&#x56F4;&#x68CB;&#x6E38;&#x620F;&#x5728;&#x7B56;&#x7565;&#x68AF;&#x5EA6;&#x6D41;&#x7A0B;&#x4E0A;&#x7684;&#x7C7B;&#x6BD4;](.gitbook/assets/leib.svg)

我们把神经网络作为围棋游戏的策略生成器，通过让两个神经网络相互下棋，来实现让智能AI学会下围棋的目的。标准围棋是19路棋盘，为了加快机器AI的学习速度，我们用9路棋盘来演示。如果读者有足够的计算机算能，可以自行调整神经网络的规模，使得模型可以学习19路棋盘的围棋下法。

![&#x56F4;&#x68CB;&#x7684;&#x7B56;&#x7565;&#x68AF;&#x5EA6;&#x7ED3;&#x6784;&#x793A;&#x610F;](.gitbook/assets/ce-lve-wang-luo-.svg)

我们用两个网络结构一样的神经网络进行对弈，并将多局对弈的过程保存成若干个SGF文件。然后再用一个专门的程序来解析这个SGF文件集合，将棋局还原成棋面与对应的着法。神经网络利用还原后的数据集合来进行训练，从而实现神经网络的下棋能力。整个过程中没有引入人为的学习样本，完全依赖策略梯度的算法教会神经网络下棋。

![](.gitbook/assets/pg-2-.svg)

1. 创建两个使用相同神经网络结构的智能体进行对弈，我们暂时称他们为智能体A和智能体B；
2. 对弈一定局数，并将棋局保存成SGF格式文件；
3. 根据对弈的结果判断是否智能体A比智能体B的水平高，如果A显著比B厉害，则用A的网络参数替换B的网络参数；
4. 解析刚才对弈的SGF文件，将棋谱还原成可以训练的样例数据；
5. 使用样例数据训练智能体A；
6. 返回第2步，轮回往复。

通过上面的流程我们就完成了一轮智能体的训练。但是一轮学习训练肯定是不够的，就拿前面那个简单的游戏来说，一般也要训练上10轮才能出现收敛的趋势，要显著收敛的话，至少也要训练100轮以上。我们的智能体则可能要训练十几万轮才能够体现出优势。但是无论训练多少轮，都是重复上面的这个过程。

策略梯度算法在神经网络的结构上没有什么特别之处，读者可以用普通的前馈网络或者卷积网络来构造自己的神经网络，不过对于初学者，建议网络不用设置的过深，能运行就好。

{% code title="myGO/utility/keras\_modal.py" %}
```python
def Model_PD(boardSize,type='C'): 
    input=keras.layers.Input(shape=(boardSize**2+1,)) 
    if type=='D':    #1
        feature=keras.layers.Dense(64*boardSize**2, kernel_initializer='random_uniform',bias_initializer='zeros',activation='tanh')(input[:,:-1])
        feature=keras.layers.Dense(32*boardSize**2, kernel_initializer='random_uniform',bias_initializer='zeros',activation='tanh')(feature)
        feature=keras.layers.Dense(32*boardSize**2, kernel_initializer='random_uniform',bias_initializer='zeros',activation='tanh')(feature)
        feature=keras.layers.Dense(16*boardSize**2, kernel_initializer='random_uniform',bias_initializer='zeros',activation='tanh')(feature)
        feature=keras.layers.Dense(16*boardSize**2, kernel_initializer='random_uniform',bias_initializer='zeros',activation='tanh')(feature)
        lnk=keras.layers.concatenate([feature, input[:,boardSize**2:boardSize**2+1]], axis=-1)
        logic = keras.layers.Dense(32*boardSize**2, kernel_initializer='random_uniform',bias_initializer='zeros',activation='tanh')(lnk)
        logic = keras.layers.Dense(64*boardSize**2, kernel_initializer='random_uniform',bias_initializer='zeros',activation='relu')(logic)
        logic = keras.layers.Dense(32*boardSize**2, kernel_initializer='random_uniform',bias_initializer='zeros',activation='relu')(logic)
        logic = keras.layers.Dense(32*boardSize**2, kernel_initializer='random_uniform',bias_initializer='zeros',activation='relu')(logic)
        logic = keras.layers.Dense(16*boardSize**2, kernel_initializer='random_uniform',bias_initializer='zeros',activation='sigmoid')(logic)
    elif type=='C':    #2
        reshape=keras.layers.Reshape((boardSize,boardSize,1))(input[:,:-1])    #3
        feature=keras.layers.Conv2D(3**4, 2, strides=1, padding='same', activation='tanh', kernel_initializer='random_uniform', bias_initializer='zeros')(reshape)
        feature=keras.layers.Conv2D(3**4, 2, strides=1, padding='valid', activation='tanh', kernel_initializer='random_uniform', bias_initializer='zeros')(feature)
        feature=keras.layers.Conv2D(3**4, 2, strides=1, padding='valid', activation='tanh', kernel_initializer='random_uniform', bias_initializer='zeros')(feature)
        feature=keras.layers.Conv2D(3**4*boardSize**2,boardSize-3,activation='tanh',kernel_initializer='random_uniform', bias_initializer='zeros')(feature)
        feature=keras.layers.Flatten()(feature)
        lnk=keras.layers.concatenate([feature, input[:,boardSize**2:boardSize**2+1]], axis=-1)
        logic = keras.layers.Dense(1024*4, kernel_initializer='random_uniform',bias_initializer='zeros',activation='tanh')(lnk)
        #logic = keras.layers.Dense(1024*2, kernel_initializer='random_uniform',bias_initializer='zeros',activation='elu')(logic)
        #logic = keras.layers.Dense(1024*2, kernel_initializer='random_uniform',bias_initializer='zeros',activation='elu')(logic)
        logic = keras.layers.Dense(512*2, kernel_initializer='random_uniform',bias_initializer='zeros',activation='relu')(logic)
        #logic = keras.layers.Dense(256, kernel_initializer='random_uniform',bias_initializer='zeros',activation='softplus')(logic)
    else:
        None
    output = keras.layers.Dense(boardSize**2,activation='softmax')(logic)
    return keras.models.Model(inputs=input, outputs=output)
```
{% endcode %}

1. 使用全连接网络来搭建神经网络结构，提取图像特征的前馈网络和负责逻辑处理的逻辑网络全部采用简单的全连接网络来实现；
2. 也可以用卷积网络来负责图像特征提取，从一些已实现的案例来看，卷积网络对图像特征识别上能工作的更好；
3. 网络的输入是将棋盘上的点平铺了，如果要使用卷积网络，需要先将平铺的输入转换成适合卷积核处理的高维度图片数据格式。

定义好了神经网络的结构，接着我们来对这五个过程一一进行说明。

第一步创建两个相同的神经网络，网络的结构读者可以自行组装，使用前面章节介绍的卷积网络或者全连接网络都可以。对于初学者而言不必将网络创建的过深，虽然可以使用残差结构缓解梯度消失的问题，但是过深的网络会需要更长的训练时间。也不一定必须要创建两个神经网络。由于围棋游戏的拟合函数比较复杂，按照神经网络的规模，网络的参数可能有上百万甚至是上千万个。使用两个神经网络进行对弈会占用大量的内存，如果是放在GPU里进行训练，一般用户的显存也不足以运行两个大规模神经网络。由于智能体A和B的网络结构是相同的，我们可以只创建一个神经网络，然后在不同的时候装载各自的参数即可。智能AI进行对弈也很简单，利用前面章节我们介绍的方法可以轻易地实现机器之间下棋。由于我们只训练智能体A，在互弈时，如果固定A一直执黑棋或者一直执白棋将会导致智能体学习到的内容有限，训练时一定会造成偏差，因此我们应当保证智能体A执黑棋和执白棋的机会均等，不要只下单边。

{% code title="/myGO/p\_g/p\_d.py" %}
```python
if __name__ == "__main__":
    pd=PD_Object()    #1
    bot1=TrainRobot(rand=bot1_isrand)    #2
    bot2=TrainRobot(rand=bot2_isrand)    #2
    bot1.dp_compile()    #3

class PD_Object():
    def make_samples(self,rounds,bot1,bot2):
        bot1_win=0
        bot2_win=0
        for i in range(rounds):
            bot1.reset()    #4
            bot2.reset()    #4
            board.reset()    #4
            game.reset(board)
            if np.random.randint(2)==0:    #5
                result=game.run_train(play_b=bot1,
                    play_w=bot2,isprint=False)    #6
                if result=='GameResult.wWin':
                    bot2_win+=1
                else:
                    bot1_win+=1
            else:
                result=game.run_train(play_b=bot2,
                    play_w=bot1,isprint=False)
                if result=='GameResult.wWin':
                    bot1_win+=1
                else:
                    bot2_win+=1
        return bot1_win,bot2_win
```
{% endcode %}

1. 创建一个策略梯度的工具包，策略梯度的对抗过程在这个工具包里实现
2. 使用相同的网络结构分别创建智能体A和智能体B
3. 我们只训练智能体A，所以只对它做训练前的配置
4. 每运行完一局游戏要将智能体和棋局的数据都重置掉
5. 50%的概率让智能体A执黑，50%的概率让它执白
6. 让智能体依据策略网络执行一局对战并返回胜负结果

对战一局最耗费时间的步骤是利用神经网络计算落子策略。对于配置不高的机器，下一局棋可能需要半分钟左右的时间。一种提高对弈效率的方法是使用多线程或者多进程的技术。但是众所周知，Python的多线程其实是一个摆设，因为Python代码的执行由Python解释器来控制，而对Python解释器的访问由全局解释器锁来控制，正是这个锁限定了解释器同时只能有一个线程在运行。很不巧，为了随大流，本书采用了Python来做代码的演示，所以我们“求之不得，寤寐思服。 悠哉悠哉，辗转反侧”，终于只剩下了使用多进程的方案，不过还有更不巧的事情，支持Windows版本的Python在多进程使用大内存的时候会存在bug，使得我们没有办法利用Python并发调用起多个Tensorflow代码，为了照顾初学者，本书的代码是在Windows下编写的，因此代码没有采用任何提高对弈效率的手段，读者可以自己尝试将代码改制到Linux平台下，利用Linux平台的Python多进程机制来提高对弈效率。我尝试了在Windows平台下通过cmd命令来调用起多个Python程序以间接实现多进程的目的，由于每局对弈都是相互独立的，没有进程间通信的需求，所以这个方案是可行的。本书的演示代码没有采用这个手段是由于这部分的工作与本书的内容没有什么关系，我想把精力尽最大努力放在原理的讲解上，读者如果有兴趣可以自己去实现这部分工作。Pytorch这个机器学习框架在学术界很是流行，我没有在Windows环境试过利用Python的多进程方法调用Pytorch，也许这个方案可行，也许不可行，这些都留给读者自己尝试吧。

把棋局保存成SGF格式是很简单的，利用我们之前介绍的sgfmill这个库就可以轻易的做到这一点。需要注意，我们要尽可能多的保存棋局信息，棋盘大小和胜负结果是一定要保存的，读者如果开发了自己的算法，还可以多保存一些其它信息，比如以多少目胜出之类的信息。SGF格式是支持自定义的，我们可以利用这一点来保存自己需要的信息。

{% code title="myGO/board\_fast.py" %}
```python
def save(self,moveHis,result):    #1
        sgftools=sgf_tools(self.board.size)    #2
        path='./lr_doc/'
        for i in moveHis:
            if i[1] is not None:    #3
                sgftools.record_game(i[0],
                    (self.board.size-i[1].row,i[1].col-1))
            else:    #3
                sgftools.record_game(i[0],i[1])
        if result==1000:    #4
            value='B+Resign'
        elif result==-1000:    #4
            value='W+Resign'
        elif result>0:    #4
            value='B+'+str(result)
        elif result<0:    #4
            value='W+'+-1*str(result)
        else:    #4
            value='0'
        sgftools.set_sgf('RE',value)    #4
        sgftools.save(path+hashlib.new('md5',
            str(moveHis).encode(encoding='utf-8')).hexdigest())    #5
```
{% endcode %}

1. 保存棋局行棋记录和胜负结果；
2. 实例化一个SGF的工具类；
3. 如果这一步不是弃手，需要对落子做额外的坐标翻译工作，原因就是SGF格式的棋盘圆点在左上角，我们的棋盘原点在左下角。如果不做映射翻译也是可以的，SGF文件保存的行棋记录就会是实际游戏一个上下翻转的镜像，但是围棋本来就是四个方向对称的，这种翻转不会对我们有什么实质性的影响。如果是弃掉一手，我们直接保存就行了；
4. 保存棋局的结果。我们先前在自定的棋盘上约定了投降返回结果是1000或者-1000。如果不是投降，实际的胜子数不会超过棋盘的大小；
5. 用棋局的MD5值给SGF格式的文件命名，这么做的一个好处是避免了（几乎不会发生）重复记录相同的棋局，另外也省去了我们考虑用什么东西来做文件名的麻烦。

经过几轮训练之后，我们需要考虑这么一个问题，经过不断迭代学习的智能体A比“老式的”智能体B强吗？显然如果A比B强，说明我们的学习训练是有效的。如果我们对弈了一局，此时A赢了，能说明A比B强吗？我们知道，对弈一局要么是A赢，要么是B赢，显然此时的胜负是含有随机性的。那么如果对弈10局呢？如果对弈了10局，A赢了6局，B赢了4局，能说明A比B强吗？就拿扔硬币来举例，扔十次，正面朝上有6次，反面朝上有4次，能说明这个硬币是不均匀的吗？为了解决这个问题，我们需要引入概率与统计方法中的基本工具之一，显著性检验。

显著性检验也称作置信度，它是为了解决真实情况与我们的假设是否有差异而诞生的，它可以作为我们对某件事情有多少信心的一种度量方式。就拿扔硬币来说，当我投掷10次硬币，其中正面朝上有6次，反面朝上有4次，根据这个情况，我能断言这枚硬币是均匀的吗？假如这枚硬币是均匀的，出现上述这种情况的概率是0.754，不高，但是也不低，不是吗？如果我抛掷100次呢？如果有60次正面朝上，出现这种情况的概率只有0.057。0.057可以算是一个小概率事件了，我就可以说我有近95%的把握认为这个硬币不够均匀。同理，放到围棋的对弈上来，如果对弈10局，A胜了B六局，我不是很有把握认为A的策略就比B好，因为如果A和B的策略水平相当，也有75%的可能出现这种情况。但是如果下了100局，其中A胜了60局，我就比较有把握认为A的策略应该比B的策略好。至于是这个小概率数字是0.057还是0.01读者可以自己定。Python中，我们可以使用Scipy库的stats.binom\_test这个工具来计算这个概率。

{% code title="/myGO/p\_g/p\_d.py" %}
```python
from scipy.stats import binom_test

if __name__ == "__main__":
    pd=PD_Object()
    bot1=TrainRobot(rand=bot1_isrand)
    bot2=TrainRobot(rand=bot2_isrand)
    bot1.dp_compile()
    for _ in range(10000):    #1
        bot1_win,bot2_win=play_against_the_other(bot1,bot2,100)    #2
        total=bot1_win+bot2_win    #1
        if binom_test(bot1_win, total, 0.5)<.05 and bot1_win/total>.5:    #3
            bot1.unload_weights(weights_old)    #4
            bot2.load_weights(weights_old)    #4
```
{% endcode %}

1. 由于不知道训练多少局能使得智能体在棋力方面有显著提升，我们简单设置一万回合先看一下训练的效果。读者也可以考虑使用`while True:`这种无限循环。经过实验，在训练1000轮后，网络就能够主动做出‘虎’的形状并叫吃了对方的子；
2. 每下100局后才进行一次训练。同时，用100局来做置信度判断可信度还是比较高的；
3. 设置0.05作为显著性判断的标准。需要注意，100局中胜利60局和胜利40局，他们的显著性检验数字都会是0.05，所以还得加上A的胜率高于B的条件。
4. 如果满足了显著性检验的条件，就可以判定训练是有效的，并且智能体A的棋力稍稍高于智能体B。我们将智能体A的网络参数装载到智能体B上，以使双方再次势均力敌。

当多局对弈完成后，对于保存的SGF文件集可使用sgfmill工具将其还原成对弈时的棋局，并把棋局的样本信息保存到HDF5文件中。读者也可以跳过保存SGF文件集这一步，直接将每一局对弈保存到HDF5文件中。选择保存成SGF文件的目的一是为了方便抽样看看智能体下棋大概是一个什么情况，二也是为了方便统一处理。故此这一步不是必须的，读者可以选择性地进行优化。

{% code title="/myGO/p\_g/p\_d.py" %}
```python
if __name__ == "__main__":
    pd=PD_Object()
    bot1=TrainRobot(rand=bot1_isrand)
    bot2=TrainRobot(rand=bot2_isrand)
    bot1.dp_compile()
    while True:
        bot1_win,bot2_win=play_against_the_other(bot1,bot2,100)
        total=bot1_win+bot2_win
        if binom_test(bot1_win, total, 0.5)<.05 and bot1_win/total>.5:
            bot1.unload_weights(weights_old)
            bot2.load_weights(weights_old)
        make_tran_data(games_doc,data_file)    #1
        games=HDF5(data_file,mode='a')    #2
        x_,y_train=games.get_dl_dset()    #3
        x_train=x_[:,:-1]    #4
        player=x_[:,-2]    #4
        winner=x_[:,-1]    #4
```
{% endcode %}

1. 每下完一百局后就将SGF文件集批量转换到HDF5文件中保存起来；
2. 实例化一个HDF5文件的工具类，用于导出训练样本与标签；
3. 100局9路围棋大约有1200回合左右，可以一次性全部载入内存；
4. 我们从HDF5里取出的样本还需稍微加工才能使用。这个加工依赖于HDF5里数据的保存方式，读者如果在HDF5里优化一下数据的组织格式，这里的加工步骤是可以省略的。

有了可以训练的数据，最后一步就是训练智能体A了。训练的方法还是基于监督学习，策略网络属于强化学习，它与传统的监督学习在训练上稍微有些不一样，差别在于对训练目标的定义不同。传统的监督式直来直去，是什么就学什么。比如我们给图像进行分类，输入一只猫的图片，我们就希望网络识别这是一只猫，输入一直狗的图片，期望网络能识别图片里有一只狗。强化学习就没这么直来直去了，由于大部分强化学习的应用场景是互动对抗，所以它引入了“价值”的概念。强化学习的样本目标不是某个具体的事物，而是输入的价值评估。

如果我们用监督学习，以9\*9路的棋盘为例，输入是棋局的当前棋形，输出是棋盘上每个点落子的概率。这些点上的概率总和等于一。这有点像图像分类，我们把9路围棋的所有可能的棋形看作一共有81种分类，然后让神经网络根据输入的棋形对其进行归类。

![](.gitbook/assets/norm.svg)

从强化学习的角度来看，81个节点的每个输出被看作是在该处落子后的获利大小。一般来说，我们把落子后的胜利看作得到了价值1，如果落子后落败，那么得到的价值是-1。但是棋类的胜负，特别是围棋，并不是由最后一步棋决定全局的结果，往往是行棋过程中的很多步棋下得都很好，逐步累积优势才得到最后的胜利。对于失败方也是一样，一定是在行棋的过程中走错了很多步棋，逐渐丧失优势才会导致最后落败。而在游戏没有结束时，对于双方的落子很难评估其好坏，只有当棋局结束时，我们才能说胜利方的一些下法是他胜利的关键，失败方有很多错误的下法，我们应该学习胜利方的下法而避免输棋者的下法。由于我们暂时没有办法为每一步都评估它的价值，于是我们可以简单粗暴地将胜者的所有下法都看作是得到了价值1，而落败方的所有下法都得到了不好的价值-1。如果只关注一局棋局，人们可能会认为这种粗暴地做法是很愚蠢的，也许胜利者的每一步下法都是好的，但是绝不能说输棋的一方没有下过一步好棋。如果我们将输棋方全部的下法都判定为错误的走法，是不是太过武断？从细节上看的确如此，不过如果我们将眼光拉的长远些，从一千局，一万局的角度来看，情况就不一样了。好的下法如果能够为胜利做出贡献，那么或多或少它会在胜方的着法中更多一些。同样道理，差的着法就会在输棋的一方中出现的更多一些。由于我们任性的断言造成的误差会在越来越多的对弈中被抹平。回忆一下我们最开始那个比大小的游戏，道理也是一样的。出5的一方也会输，原因只是他出5出的比胜利的一方少。如果增加胜利者出5的概率，同时减少输家出5这张牌的概率，综合起来看，我们还是增加了出5的概率的。

![](.gitbook/assets/vadl.svg)

在很多对强化学习介绍的材料里，我们常常会看到对价值进行”折现“。这是一个从经济学里引入的概念，比如我存了银行3年定期，到期了我拿出了1万元，折现的概念就是说对于这1万元，1年前是多少钱，2年前是多少钱，最初存入的是多少钱。以此类比我们强化学习中价值的概念，胜利后得到了价值1，那么在前一步落子时是多少价值，再往前一步的落子是多少价值。折现的概念我们就这样简单介绍一下完事，对于围棋游戏，我们不对价值进行折现，原因前面也说了，最关键的那一步往往不是最后那步落子。

对于分类问题或者多选项问题，神经网络最后的激活层通常有两类选择，一类是每个输出节点单独使用sigmoid或者tanh函数作为激活函数，另外一类则是使用softmax函数作为激活函数。对于第一类激活函数，其输出的数值总和绝大多数情况下都不是一，而且对其输出的值也未必一定要解释为概率。对于第二类激活函数，其输出的数字总和总是一，因此可以将其解释为对各个输出节点的概率判断。第一类激活函数和第二类激活函数有时候可以混用，有时候则不可以。比如对MNIST手写数字进行识别，我们就可以在神经网络的最后一层使用softmax函数作为激活函数，原因是数字总是0到9中的一个，如果我们知道被识别的对象是某个数字，那么总要给被识别对象分配一个结果。但是如果是做一个图像分类，那么使用第一类的sigmoid函数作为激活层会更合适，因为总有我们的神经网络没有见过的图片类别，它不能被归类到已知的任何类别中去，使用softmax函数会强制将其归到某一个已知类别中，这就限制了网络泛化的能力。

对于围棋AI的学习，我们选择使用softmax作为最后输出层的激活函数。主要是考虑到使用softmax学习负样本的损失函数实现起来非常方便。多分类的损失函数是：

$$
L(x_i)= -\sum_k y\cdot log \hat{f}(x_i)
$$

其中y是训练样本的标签，f\(x\)是网络的实际输出。

当样本的下法是该棋局胜方的落子时，表示这个下法是我们期望学习到的内容时，我们设置训练样本对应的标签为1，这时就和普通的分类学习一样。但是当样本的下法属于输棋的一方时，我们认为这一步不值得学习，它的价值为负，策略网络应该避免下出这一步，我们希望策略网络的输出尽量偏离这个训练样本的标签，即策略网络往正样本学习的反方向变动，这时我们通过设置y等于-1即可实现这个小目标。此时损失函数的作用由原本期望标签与估计差距尽可能小变成了两者的差距尽可能的最大。

$$
-\sum_k (-y)\cdot log \hat{f}(x_i)=\sum_k (y)\cdot log \hat{f}(x_i)=-L(x_i)
$$

如果使用第一类函数，我们需要对每个节点做单独的二分类计算。二分类的损失函数是：

$$
L(x_i)=-(y\cdot\log(\hat{y})+(1-y)\cdot\log(1-\hat{y}))
$$

显然，如果使用第一类函数作为激活函数，我们不能通过简单地将样本标签y设置为-1来实现网络的反向偏转。

$$
-((-y)\cdot\log(\hat{y})+(1-(-y))\cdot\log(1-\hat{y}))=y\cdot\log(\hat{y})-(1+y)\cdot\log(1-\hat{y})\neq-L(x_i)
$$

我们只能手工编写自定义的损失函数，这对于初学者来说有点复杂，也有点偏离了我们制作一款超越人类水平的围棋软件这个主题。如果读者有兴趣使用第一类激活层，可以尝试编写自己的损失函数，并比较一下两种激活层是否会在学习效率上有差异。Keras允许用户自定义损失函数，下面这个例子演示了如何在Keras里使用自定义损失函数，示例的方法和我们直接使用正负1作为样本标签的效果是一样的。

```python
import keras.backend as K

#Returns则是价值，正向价值是1，负价值等于-1
def policy_gradient_loss(Returns):
    #action是输入，action_probs是输入对应的输出        
    def modified_crossentropy(action,action_probs):    
        cost = K.categorical_crossentropy(action,
            action_probs,from_logits=False,axis=1 * Returns)    
        return K.mean(cost)
    return modified_crossentropy
```

读者需要注意，我在实际的代码里使用softmax作为激活函数与前面的说法存在一点逻辑上的分歧。通常在强化学习中，我们把每个输出看作是对行棋后的价值估算，好的价值是1，坏的价值是-1。但是softmax本身并不能输出小于0的数值，从这个角度来看，似乎用tanh来作为最后一层网络的激活函数更妥当一些。这里我们稍微要做一些变通。当我们把神经网络作为我们下棋的策略时，不管是把输出看作是每个点落子的概率也好，或者是每个点落子的价值也好，在最终选择时我们都只选择那个值最大的作为网络给我们的落子建议。从这个角度来看，我们只希望最佳的落子选择节点能够输出尽量大的值，从而保证我们能够选到它，至于其它的落子位置，他们输出的是-1还是0反而显得就不那么重要了。选用softmax完全可以保证这一点，它能够做到最大化我们希望的节点输出，同时压抑其它我们不需要的节点输出。如果我们认为失败的下法没有价值，将坏价值设置为0，就可以统一两种不同逻辑的数学形式了。

{% code title="/myGO/p\_g/p\_d.py" %}
```python
if __name__ == "__main__":
    ...
    while True:
        ...
        x_train=x_[:,:-1]
        player=x_[:,-2]
        winner=x_[:,-1]
        for  i,y in enumerate(player==winner):    #1
            if y==False:    #1
                y_train[i][y_train[i]==1]=-1    #1
        bot1.dp_fit(x_train,y_train)    #2
        bot1.unload_weights(weights_current)    #3
```
{% endcode %}

1. 为了记录着法，样本标签中将落子位设置成1，其它位置都是0。我们使用策略梯度时，希望避免输棋方的下法，所以需要将输棋方的落子设置成-1，赢棋方的落子标签保持1不变；
2. 设置好样本和它对应的标签后，训练过程就和普通的监督学习没有什么差别；
3. 每训练完一轮后，我们需要保存下当前网络的权值。在下一轮训练时，如果发现智能体A显著强于智能体B时，我们就要那这个保存的权值去替换掉智能体B的网络权值。

在训练时我们还需要谨慎地选择网络输出的值，如果仅简单粗暴地选择最大输出，那么不论下多少回合，结果都是一样的。因为当神经网络的输入和参数都固定的时候，它的输出也一定是确定的。当然我们可以每下一局学习一次，但是这种办法会使得训练过程不够稳定，相关性太强的数据容易使得训练结果无法收敛。如何使得我们的网络输出能够有多样性呢？我们可以参照前面提到的卡牌小游戏，我们可以将策略网络的输出看作是我们最终取该节点作为落子点的概率，不是简单地取最大值，而是依据概率来取值，从而保证了相同的策略网络参数下多局也不会产生一样的棋局。Python中有`random.choices`这个工具，可以方便地实现这个功能。

{% code title="/myGO/p\_g/p\_d.py" %}
```python
class TrainRobot():
    ...    
    def predict(self,player,board,reChoose=False,isRandom=None): 
        if not reChoose:    #1
            self.moves=frozenset()
        if not np.random.randint(100):    #2
            rand=True
        if rand:    #3
            ...
        else:
            while True:
                ...
                pred=self.model.model_predict(npdata)    #4
                pred_valid=mask*pred    #5
                pick=random.choices(range(boardSize*boardSize),
                    weights=pred_valid.flatten())     #6
                move=np.unravel_index(pick[0], (boardSize,boardSize))    #6
                self.moves=self.moves|{move}    #7
                point=Point(move[0]+1,move[1]+1)
                if self.isPolicyLegal(point,board,player):    #8
                    return Move(point=point)
                else:
                    continue
```
{% endcode %}

1. 由于全局同形的判断没有放在智能体的方法类中，所以一旦棋局判断发生了全局同形，我们只能要求智能体重新下一步。这里提供一个类似记忆库的功能，让智能体记住最近下过哪些着法；
2. 为了避免由于策略网络的参数不够好，导致网络在产生数据的时候存在偏差，我们在网络对弈时引入一定比例的随机落子，比如这里以百分之一的概率选择随机策略；
3. 随机策略的实现比较简单，读者可以自行查阅源码，这里不进行累述；
4. npdata是棋盘数字编码后的结果，是一个nump结构，我们将其输入网络，得到网络对每个落子点的概率判断；
5. mask的作用是过滤掉非法落子点，一般情况下，当神经网络学习到一定程度就不应该再输出非法落子了，但是由于我们的选择策略不是取最优值，而是依据概率来取值，所以屏蔽掉非法着法就成了必不可少的一步；
6. 我们使用random.choices来实现依据概率来选择着法这件事；
7. 将选择的着法保存到最近落子的记忆库中，如果碰到了全局同形或者策略违法（不能自杀等），就能利用mask变量屏蔽部分策略的功能来避免重复落子；
8. 判断一下当前的策略给出的建议落子是否违反了我们自己的策略，如果违反了就重新再选择着法。

我们在制作学习样本的时候简单地将赢棋一方的落子都判定为好的下法，输棋的一方落子都是差的下法。根据前面的描述我们知道这样做是可行的。不过即便我们对此有信心，单凭赢棋方的一步好棋在完整的一局棋里能发挥的作用还是有限的。围棋中，吃掉对方一粒棋子可能会带来优势，但这并不会对赢棋起决定性的作用。更何况还有很大可能，我们学习的一步还是一步差棋。这个道理对于输棋一方也是一样的，也许被我们抛弃的一步棋其实是一步好棋，正是这步好棋避免了输的一方输的更惨一些。因此，我们在神经网络的学习率这个超参上要比以往的模型要更谨慎一些。也许学习率等于0.1并不会让网络的学习过程发散，但是可能0.1这个学习率太高了，我们不想单凭一步的学习就让整个网络产生较大变化，所以也许0.000001会是一个适当的学习率。不过读者也可以根据自己的理解调整这个参数，但是不要太大这一点应该是不错的。基于同样的道理，我们对对弈产生的样本只学习一次，不像标准的监督学习，样本可以反复使用。通过智能体间不断地对弈，理论上，如果整个训练过程适当，智能体间对弈产生的样本质量会越来越高，伴之我们智能体的棋力也越来越强。

```python
class DenseModel():
    ...
    def p_d_compile(self):
        self.model.compile(optimizer=
            keras.optimizers.SGD(learning_rate=0.0001),    #1
            loss=keras.losses.CategoricalCrossentropy(from_logits=False),
            metrics=['accuracy'])
    def fit_all_data(self,x,y,batch_size=512,
        epochs=1,earlystop=0,checkpoint=False):    #2
        ...
        self.model.fit(x,y, batch_size=batch_size,
            epochs=epochs,callbacks=callbacks)
```

1. 为了提高梯度下降算法的速度与效率，人们发明了很多优秀的优化算法，比如Adam，RMSprop等。但是这些优化算法在设计的时候考虑的对象是传统的监督学习。强化学习能够自己产生无穷无尽的学习样本和标签，而且这些学习素材往往只会使用一次就抛弃了。传统的监督学习样本却需要反复使用。因此这些针对传统监督学习的优化算法也许并不适合继续在强化学习中使用。当然很可能在99%的情况下使用这些优化算法可以让我们得到理想的结果，但是使用传统的梯度下降算法更能够保证理论上的一致性，避免发生一些意料之外的情况；
2. 由于我们有足够多的样本可以用来学习，而且我们对单个样本是否真的有效并不报太大希望，所以所有的样本我们只学习一次。经验上看，对于围棋的强化学习，每一轮训练1000个左右的样本能够得到理想的结果。

我们开源的代码里用梯度下降法训练了9路围棋的智能程序，在训练一千轮后，我们的智能体就能够主动发起叫吃并在对手不应后吃掉对方的子。9路围棋因为棋盘不大，吃掉对方的子将会带来巨大地优势，所以神经网络会发现这个下法并不令人吃惊。

