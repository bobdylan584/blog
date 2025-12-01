---
date: 2025-12-01
title: Transformer作者爆料GPT-5.1内幕！OpenAI内部命名规则变乱了
categories: [NEWS, Transformer作者爆料GPT-5.1内幕！OpenAI内部命名规则变乱了]
tag: NEWS
---

- [首页](http://www.qbitai.com/)
- [资讯](https://www.qbitai.com/category/%e8%b5%84%e8%ae%af)
- [数码](https://www.qbitai.com/category/ebandeng)
- [智能车](https://www.qbitai.com/category/auto)
- [智库](https://www.qbitai.com/category/zhiku)
- [活动](https://www.qbitai.com/category/huodong)
- [MEET大会](https://www.qbitai.com/meet/meet2025/)
- [AIGC](https://www.qbitai.com/meet/aigc2025/)









# Transformer作者爆料GPT-5.1内幕！OpenAI内部命名规则变乱了

![img](http://www.qbitai.com/wp-content/themes/liangziwei/imagesnew/head.jpg)*衡宇* 2025-11-30 13:18:27 来源：[量子位](http://www.qbitai.com/)

下一次突破来自多模态推理+具身智能

> 鹭羽 发自 凹非寺
>
> 量子位 | 公众号 QbitAI

> 我们正在经历一次静悄悄、但本质性的AI范式转换。
>
> 它的意义不亚于Transformer本身。

过去一年里，关于**AI发展**，出现了两种观点的分化：

- 一边是“AI增长放缓、模型到顶、预训练无用论”
- 另一边则是隔三差五就来一次“AI大周”：**GPT-5.1**、**Gemini 3**、**Grok 4.1**。

![Transformer作者爆料GPT-5.1内幕！OpenAI内部命名规则变乱了](https://www.qbitai.com/wp-content/uploads/replace/039488fd0c71ee051dfc6a52021b6be6.png)

而Transformer作者之一、现任OpenAI研究科学家的**Łukasz Kaiser**最近接受采访，给出了第一视角的解答。

信息量极大，包括AI的底层范式转变、GPT-5.1的命名规则、未来AI的发展趋势……以及**Transformer**诞生背后的二三事。

> AI不是变慢了，而是换代了。
>
> GPT-5.1不是简单的小版本迭代，OpenAI内部版本命名规则有所变化。
>
> 多模态推理将会成为下一个突破点。
>
> AI不会让人类完全失去工作。
>
> 家用机器人是继ChatGPT后最可见的AI革命。

下面一起来康康详细内容：

# AI发展没有放缓，而是平稳增长

过去一年里，有关“模型进展变缓”的声音层出不穷，但Łukasz认为这种看法是错误的。

![Transformer作者爆料GPT-5.1内幕！OpenAI内部命名规则变乱了](https://www.qbitai.com/wp-content/uploads/replace/1dcffa30340904ac45ae3205055e141d.png)

他给出的解释也很直白：

> 从内部视角看，AI的能力增长是一条非常平滑的指数曲线。

这类似于**摩尔定律**，几十年来摩尔定律始终有效，甚至在GPU的推动下还在加速，归根结底也是因为它历经了数代技术的迭代。

因此，AI从外部看，趋势是平稳的；而从内部看，其进步也离不开新技术、计算机能力的提升和工程优化的共同作用。

至于为什么会有人觉得“变慢了”，原因无它：**AI的底层范式，已经悄悄从预训练转向推理模型。**

这也是继Transformer诞生后的又一次关键转折。

如果把技术发展的过程描述为一条S型曲线（起步→快速增长→平稳期），那么预训练就处于S曲线的上升后期，而推理模型仍处于初期。

不过这并不意味着预训练的**Scaling Laws**就失效了，它仍在发挥作用，只是和新的推理范式相比，需要投入更多的资金。

![Transformer作者爆料GPT-5.1内幕！OpenAI内部命名规则变乱了](https://www.qbitai.com/wp-content/uploads/replace/61c345623f8631aa6bfebe4be4493910.png)

所以出于经济上的考量，业内人士开始普遍将工作重心转向更小也更便宜，但质量相同的模型，所以这也是导致外界认为预训练已经停止的原因之一。

那么回到推理模型上，由于该范式还处于新兴阶段，进步速度会相当之快。

以**ChatGPT**为例，GPT-3.5会直接基于训练数据记忆给出答案，而不会借助任何外部工具和推理，反观现在最新的ChatGPT会主动浏览网站、进行推理分析，再给出准确答案。

对于普通用户来说，如果不仔细对比，可能会觉得二者差异不大，但实际上这背后是性能质的飞跃。

又比如说**Codex**，程序员的工作方式已经在近几个月里转变为**“Codex先处理，然后人工微调”**的模式，这种变化其实相当之彻底，但如果不是专业从事编程工作，自然不会留意到这种根本性变革。

![Transformer作者爆料GPT-5.1内幕！OpenAI内部命名规则变乱了](https://www.qbitai.com/wp-content/uploads/replace/9d1127926e44362d29bd2e66ebf55cf5.png)

所以总的来说，这一切的变化都发生得太快，以至于让人们还未曾察觉到其中的变化。

而推理模型的本质其实也与基础大模型类似，只是在给出最终答案前，会优先进行思考，也就是所谓的**思维链**。

在思考过程中，模型被允许使用工具，例如浏览网页，以给出更准确的答案。其推理过程也会被视为模型的一部分并接受训练。

相比于传统的深度神经网络梯度下降训练，推理模型则更多使用的是**强化学习**。

具体来说，强化学习会通过奖励机制推动模型获取更好的答案，也需要研究人员提供更细致的数据准备，以完成强化学习的参数调整。

然后通过强化学习，模型就能学会对自身错误的纠正。

后续行业也会继续转向更复杂的强化学习，例如借助一个大模型来判断答案的正确性或偏好度，或者融入更多的人类偏好。

![Transformer作者爆料GPT-5.1内幕！OpenAI内部命名规则变乱了](https://www.qbitai.com/wp-content/uploads/replace/ec6f70236177e093b1006ed0402d4c2b.png)

总之，未来强化学习的应用范围会更加广泛，不仅仅适用于特定领域，还能处理更多通用数据，比如说多模态推理，虽然最近**Gemini**已经能够在推理过程中生成图像，但整体来说还处于刚刚起步的阶段，相信在强化学习的帮助下会有进一步的提升。

# GPT-5.1绝非表面上的小版本更新

关于最近发布的**GPT-5.1**，Łukasz也释出了更多细节。

> GPT-5.1看起来只是小版本更迭，实际从内部来讲，是一个巨大的稳定性迭代。

首先回到最初的GPT-4到GPT-5，简单来说，得益于强化学习和合成数据的应用，GPT-5的推理能力明显提升了。

而到GPT-5.1的改进，则更多集中在**后训练**阶段，比如增加安全性、减少幻觉，以及添加了如书呆子、专业等多种风格选择。

![Transformer作者爆料GPT-5.1内幕！OpenAI内部命名规则变乱了](https://www.qbitai.com/wp-content/uploads/replace/ffee7de09c4a57614c3701e0a9d64f8e.png)

版本的命名方式也不再与技术细节挂钩，转而**以用户体验为导向**，比如GPT-5是基础能力较强的模型，GPT-5.1是能力更优的版本，Mini是更小、更快、更廉价但性能稍弱的模型，推理模型则专注于复杂任务。

这种命名方式的转变也为OpenAI内部带来了更多灵活性，现在强化学习、预训练、幻灯片优化等多个项目并行工作，然后通过蒸馏技术就能将多项目成果整合到一个模型中。

这大大缩短了模型迭代时间，可以更好地满足用户体验需求，所以GPT-5.1看似是小版本更新，实则背后是OpenAI基于用户对其能力和目标预期做出的策略调整。

不过坦白地讲，GPT-5.1在部分能力上仍然存在短板。

![Transformer作者爆料GPT-5.1内幕！OpenAI内部命名规则变乱了](https://www.qbitai.com/wp-content/uploads/replace/6e4437de403d6188b09a327244b2b44a.png)

比如Łukasz用自己5岁的女儿举了个例子——

GPT-5.1能够游刃有余地解决奥林匹克竞赛题，但在面对小学一年级的**数奇偶数题目**上却错误百出。

该题目内容是，图中有两组点，中间有一个共享点，问总点数是奇数还是偶数。

5岁的孩子能够在10秒内就算出答案（因为共享点的存在导致总点数为奇数），但无论GPT-5.1还是Gemini 3都会自动忽略这个共享点，误判为偶数。

这主要还是因为模型缺乏足够的多模态能力，也未能将一个问题的推理经验迁移到相似场景中，所以后续他们将会在训练中进一步强化多模态推理和上下文推理迁移能力。

# 从谷歌Transformer走向OpenAI

而作为Transformer的作者之一，Łukasz也在访谈中补充了很多诞生细节。

![Transformer作者爆料GPT-5.1内幕！OpenAI内部命名规则变乱了](https://www.qbitai.com/wp-content/uploads/replace/d7594ca1519774b98e84641d00ea53ab.jpeg)

Łukasz自己原先是一名专注于理论计算机科学的学者，高中时就对数学和计算机充满兴趣，并在德国获得了理论计算机科学与数学博士学位。

他一直对“思维是如何运作的”、“智能的本质是什么”诸如此类的问题充满好奇，也曾在法国获得终身教职，从事逻辑和编程研究。

直到深度学习兴起，他加入了**谷歌**。

先是成为了Ray Kurzweil团队的一员，后转至Google Brain，开始与**Ilya Sutskever**等人合作。

在开发Transformer的过程中，Łukasz主要负责编码和系统工作，参与TensorFlow框架的开发。

![Transformer作者爆料GPT-5.1内幕！OpenAI内部命名规则变乱了](https://www.qbitai.com/wp-content/uploads/replace/d5f43352129abd526f3b33cab42f4dd2.png)

不过有趣的是，据他回忆，Transformer论文的八位共同作者**从未在同一个物理房间中共同出现过**。

而虽然他们彼此之间素未谋面，但他们通过不同角度共同构建了这个模型：

有人专注于注意力机制本身，有人研究如何通过前馈网络存储知识，还有人复杂解决工程实现问题，比如他自己。

从现在的角度看，Transformer毫无疑问是当今AI架构的里程碑，但在当时，很多人对用同一个模型处理多个任务的想法并不理解，他们普遍认为不同任务就应该分别训练不同的专有模型。

而他们八个人坚信自己的选择，后来的事实也证实了他们的想法是正确的。

![Transformer作者爆料GPT-5.1内幕！OpenAI内部命名规则变乱了](https://www.qbitai.com/wp-content/uploads/replace/847d6bbc8c6f8eb20f9ef845bf85a905.png)

关于之所以离开谷歌，转投**OpenAI**，其中一个原因还是因为llya。

llya在谷歌时期就是Łukasz的直系领导，在创办OpenAI后也屡次邀请他加入。刚好这时，Łukasz也无法适应Google Brain的团队规模扩大以及远程工作氛围，于是一拍即合，来到了OpenAI。

OpenAI也没有让他失望，这里没有严格的组织架构，都是根据项目自发组队，也会根据项目进展灵活调整，直到项目成熟才会逐步扩大团队。

当然不同项目之间也会存在资源竞争，毕竟**OpenAI内部GPU资源有限**。

从技术层面看，预训练目前消耗的GPU资源最多，其次是强化学习和视频模型，资源分配在很大程度上还是由技术需求决定。

所以竞争不可避免，Łukasz本人也不例外。

![Transformer作者爆料GPT-5.1内幕！OpenAI内部命名规则变乱了](https://www.qbitai.com/wp-content/uploads/replace/a0d4695325af5fd5b718c9aab2819af0.png)

# 下一次突破来自多模态推理+具身智能

最后，Łukasz聊了聊他眼中的AI未来。

> AI会改变工作，但不会让工作消失。

因为从产品层面上看，即使AI自动化了绝大部分任务，但**人类专家的需求仍然存在**。

以翻译行业为例，其实Transformer论文最初的应用场景就是翻译，现在的模型也能准确翻译西班牙语、法语等语言，但对于报纸广告乃至ChatGPT UI界面，仍然需要人类译者进行二次审核。

![Transformer作者爆料GPT-5.1内幕！OpenAI内部命名规则变乱了](https://www.qbitai.com/wp-content/uploads/replace/602e6ce43f99069fa8e517f545063c39.png)

这本质上是信任问题，即使模型能力再强，对于一些高风险、高关注度的场景，还是会倾向于依赖人类专家经验。

只是说，对于另外一些基础工作，可替代性会变高，后续也会出现相应的工作内容变化，但归根结底不会让人类无事可做。

Łukasz还预计，**家用机器人**可能会成为“下一次更为直观的AI革命”。

![Transformer作者爆料GPT-5.1内幕！OpenAI内部命名规则变乱了](https://www.qbitai.com/wp-content/uploads/replace/aa4c4bd088b4e0bdbe9a57943da8a562.png)

机器人技术的进展，取决于多模态能力和通用强化学习、通用推理的进步。一旦这些领域取得突破，机器人技术必将迎来爆发式增长。

目前已经有很多硅谷公司在相继推出智能手遥操作等硬件产品，硬件基础也将迅速成熟，届时将协同多模态和物理世界推理能力，实现家用机器人的能力跃迁。

这将会比ChatGPT**更直观、更易感知**。

参考链接：
[1]https://www.youtube.com/watch?v=3K-R4yVjJfU&t=2637s

*版权所有，未经授权不得以任何形式转载及使用，违者必究。*

[GPT-5.1](https://www.qbitai.com/tag/gpt-5-1)[OpenAI](https://www.qbitai.com/tag/openai)[Transformer](https://www.qbitai.com/tag/transformer)



[![img](http://www.qbitai.com/wp-content/themes/liangziwei/imagesnew/head.jpg)](https://www.qbitai.com/2025/11/357506.html#)[衡宇](https://www.qbitai.com/author/hengyu)

- [苹果AI论文太坑了！用GPT写的GT，导致北京程序员通宵加班](https://www.qbitai.com/2025/11/357282.html)*2025-11-28*
- [阶跃开源4B Agent模型，跑通所有安卓设备，手搓党一键部署](https://www.qbitai.com/2025/11/357627.html)*2025-11-30*
- [国产家庭机器人终于落地！连人带床推你去上班，小五位数价格明年开卖](https://www.qbitai.com/2025/11/357263.html)*2025-11-28*
- [居然有21%的ICLR 2026评审纯用AI生成…](https://www.qbitai.com/2025/11/357579.html)*2025-11-30*











### 相关阅读

- [![img](https://www.qbitai.com/wp-content/uploads/2024/04/DALL%C2%B7E-2024-04-21-11.56.31-A-cartoon-image-of-a-hacker-working-on-a-laptop-in-a-dark-room-surrounded-by-digital-screens-displaying-code-and-cyber-security-symbols.-The-hacker-i-e1713676072585.jpg)](https://www.qbitai.com/2024/04/136662.html)

- 

- 

- #### [GPT-4化身黑客搞破坏，成功率87%！OpenAI要求保密提示词，网友复现ing](https://www.qbitai.com/2024/04/136662.html)

- 单次成本60块

- [西风](https://www.qbitai.com/?author=47833)

- 2024-04-21

-  [GPT-4](https://www.qbitai.com/tag/gpt-4) [OpenAI](https://www.qbitai.com/tag/openai) [黑客](https://www.qbitai.com/tag/%e9%bb%91%e5%ae%a2)

- [![img](https://www.qbitai.com/wp-content/uploads/replace/2fff416de2c93c65b5c99afe0c58a81f-e1695288593890-300x175.png)](https://www.qbitai.com/2023/09/85355.html)

- 

- 

- #### [OpenAI公开招人！骗骗大模型就给钱，一年只工作5-10小时，无需技术背景](https://www.qbitai.com/2023/09/85355.html)

- 还有机会和各界大佬交流

- [克雷西](https://www.qbitai.com/?author=47832)

- 2023-09-21

-  [OpenAI](https://www.qbitai.com/tag/openai)

- [![img](https://www.qbitai.com/wp-content/uploads/2024/05/2024-05-17-09.03.59-e1715908070851.png)](https://www.qbitai.com/2024/05/143068.html)

- 

- 

- #### [ChatGPT接力更新：实时交互式分析Excel！网友扒出背后新模型](https://www.qbitai.com/2024/05/143068.html)

- 数据分析神器，打工人狂喜

- [西风](https://www.qbitai.com/?author=47833)

- 2024-05-17

-  [OpenAI](https://www.qbitai.com/tag/openai) [数据分析](https://www.qbitai.com/tag/%e6%95%b0%e6%8d%ae%e5%88%86%e6%9e%90)

- [![img](https://www.qbitai.com/wp-content/uploads/2025/08/-e1756358852707-300x175.jpg)](https://www.qbitai.com/2025/08/326827.html)

- 

- 

- #### [OpenAI和Anthropic罕见互评模型：Claude幻觉明显要低](https://www.qbitai.com/2025/08/326827.html)

- 网友：以后就这么搞！

- [十三](https://www.qbitai.com/?author=21)

- 2025-08-28

-  [Anthropic](https://www.qbitai.com/tag/anthropic) [Claude](https://www.qbitai.com/tag/claude) [GPT-4](https://www.qbitai.com/tag/gpt-4) [OpenAI](https://www.qbitai.com/tag/openai)

- [![img](https://www.qbitai.com/wp-content/uploads/2021/09/bert-ernie1.jpeg)](https://www.qbitai.com/2021/09/28424.html)

- 

- 

- #### [清华唐杰团队：一文看懂NLP预训练模型前世今生](https://www.qbitai.com/2021/09/28424.html)

- 

- [博雯](https://www.qbitai.com/?author=37)

- 2021-09-08

-  [BERT](https://www.qbitai.com/tag/bert) [GPT](https://www.qbitai.com/tag/gpt) [Transformer](https://www.qbitai.com/tag/transformer) [唐杰](https://www.qbitai.com/tag/%e5%94%90%e6%9d%b0) [迁移学习](https://www.qbitai.com/tag/%e8%bf%81%e7%a7%bb%e5%ad%a6%e4%b9%a0) [预训练模型](https://www.qbitai.com/tag/%e9%a2%84%e8%ae%ad%e7%bb%83%e6%a8%a1%e5%9e%8b)

- [![img](https://www.qbitai.com/wp-content/uploads/2023/01/GPTZero%E5%B0%81%E9%9D%A2-e1675147200797-300x175.jpeg)](https://www.qbitai.com/2023/01/41707.html)

- 

- 

- #### [「ChatGPT克星」升级：老师可以把全班作业丢进去检测了！华人作者：免费用](https://www.qbitai.com/2023/01/41707.html)

- 支持Word、PDF，开放API

- [十三](https://www.qbitai.com/?author=21)

- 2023-01-31

-  [ChatGPT](https://www.qbitai.com/tag/chatgpt) [GPTZero](https://www.qbitai.com/tag/gptzero) [OpenAI](https://www.qbitai.com/tag/openai) [人工智能](https://www.qbitai.com/tag/%e4%ba%ba%e5%b7%a5%e6%99%ba%e8%83%bd)

### 热门文章

![img](https://www.qbitai.com/wp-content/uploads/2025/11/2025-11-24-15.06.41-e1763968044895-150x150.png)

#### 上线4天下载破百万，蚂蚁CTO：灵光要做AGI时代的“支付宝”

2025-11-24

![img](https://www.qbitai.com/wp-content/uploads/2025/11/61cc0918403fa5644b7778bce4bd4020e8aef7cd-1920x1440-e1764149308179-150x150.jpg)

#### 开源模型叫板Nano Banana Pro！Stable Diffusion原班人马杀回来了

2025-11-26

![img](https://www.qbitai.com/wp-content/uploads/2025/11/2025-11-25-09.05.11-e1764034228772-150x150.png)

#### Claude Opus 4.5发布！2小时工程测试超人类，前代Sonnet搞不定的活它轻松拿捏

2025-11-25

![img](https://www.qbitai.com/wp-content/uploads/replace/049c1f14de53576f687dda4c00343f7e-e1764142025960-150x150.png)

#### 超越Gemini3、GPT5.1，阿里千问登顶空间推理全球冠军

2025-11-26

![img](https://www.qbitai.com/wp-content/uploads/2025/11/04c762108ff3f1e85c4efd0dfaa02342-150x150.jpg)

#### 国内最大AI“学术-产业-人才”盛会来了！20位院士+50位院长+300位专家集结北京海淀

2025-11-26

- [关于量子位](https://www.qbitai.com/?page_id=99)
- [加入我们](https://www.qbitai.com/?page_id=101)
- [寻求报道](https://www.qbitai.com/?page_id=103)
- [商务合作](https://www.qbitai.com/?page_id=105)



 



 



 



追踪人工智能新趋势，报道科技行业新突破



量子位 QbitAI 版权所有©北京极客伙伴科技有限公司 [京ICP备17005886号-1](https://beian.miit.gov.cn/)