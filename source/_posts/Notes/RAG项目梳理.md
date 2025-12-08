---
date: 2025-04-05 00:27:31
title: RAG项目梳理
categories: [AI, RAG项目梳理]
tag: AI
---

# 目前代码用到的包

pymilvus、logging

# 模块一、听课笔记

## 项目背景

### 能解决大模型幻觉、过时、安全、专业知识的弊端的解决方法就是：RAG+微调。

### RAG和微调本质上都是为了，让大模型更好地基于企业私有知识来回答问题，RAG提供外部知识库，提供数据来源。微调基于知识库，告诉大模型怎么生成答案。例如提示词微调Prompt 微调。

### 基于企业私有知识的垂直领域微调：拿企业的知识做微调训练比如PEFT中lora拿少量数据集，去给大模型做一个微调。

### 基于企业私有知识的RAG问答系统：给企业的私有知识创建一个数据库，让大模型去做意图识别，从而进行专有领域的一个问答。

### 微调的弊端：

需要微调人才+需要显卡资源

### 通用模型和垂直领域模型的区别：

能做所有任务，通晓所有领域常识的模型就是通用型或者说通用领域的大模型，专精于某个领域（e.g.法律）的就是垂直领域大模型。

### 项目流程

**不进入RAG**:用户输入query，先经过mysql检索阈值>=0.85的高频回答对。存在直接返回，无需进入RAG流程。

进入RAG：找到本地文档--文档切分，切分的文档做embedding，存入milvus。

query也做embedding，去milvus数据库检索，得到topk，topk和query进行拼接，让我模型去思考生成结果。



文档切分：因为query太短，embedding的文章太长，一定要切分。

## 项目工具介绍

### langchain的主要组件

Models：chat_models、llms、embedding_models

from langchain_community.llms import Ollama

from langchain_community.chat_models import ChatOllama

```
llms:生成式语言模型：
from langchain_community.llms import Ollama
model = Ollama(model='qwen2:1.5b',temperature=0)
model.invoke('请给我讲个故事吧')
temperature=0；代表严谨程度，温度越冷，就代表越严格挑选概率最高的词生成。通常用于代码生成。温度越高，就代表例如=1，就代表概率极低的词也有可能被生成，通常用于创作任务。
```

```
chat_model:聊天模型
from langchain_core.message import HumanMessage, SystemMessage
from langchain_community.chat_models import ChatOllama
model = ChatOllama(model='qwen2:1.5b',temperature=0)
messages = [
        SystemMessage(content="现在你是一个著名的歌手"),
        HumanMessage(content="给我写一首歌词")
]
res = model(messages)
```

```
OllamaEmbeddings：文本嵌入模型
from langchain_community.embeddings import OllamaEmbeddings
model = OllamaEmbeddings(model='mxbai-embed-large',temperature=0)
接收一个字符串的输入
res1 = model.embed_query('这是第一个文档')
接收多个字符串的输入
res2 = model.embed_documents(['这是第一个测试文档','这是第二个测试文档'])

```

Prompt：

```
from langchain import PromptTemplate
from langchain import PromptTemplate, FewShotPromptTemplate
```

Memory、

```
from langchain.memory import ChatMessageHistory
from langchain import ConversationChain
#  实例化大模型
llm = Ollama(model="qwen2.5:7b")
conversation = ConversationChain(llm=llm)
```

Indexes、

Chains、

```
from langchain.chains import LLMChain
```

Agents

```
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.agents import get_all_tool_names
```

## milvus 向量数据库

### WeightedRanker加权排名策略

#### 定义：

是一种融合一条实体（样本）多个向量字段检索结果评分（score）的方法。核心是根据不同向量字段的重要性，为各路搜索结果的得分分配权重，再将这些加权得分融合，得出最终排序。

#### 其原理和步骤：

**1、收集搜索得分**

从不同的向量检索路径，分别收集返回的结果及其得分（score1、score2）

**2、得分归一化**

不同路径的得分范围通常不同（如内积IP可能是[-∞，+∞]，并且值越大越匹配，欧式距离L2是[0，+∞），并且值越小越匹配，直接比较不合适，通过对分数使用如函数等数学变换，将各得分统一映射到(0,1]区间，方便统一秩序比较。例如欧式L2的值越小，就映射越接近1。IP和COSINE值越大就，映射成分数越接近于1。

**3、分配权重。**

根据不同向量字段的重要性，为每一个路径的归一化得分分配权重，权重值在0-1之间。例如image_recognization_vector字段权重为0.6、txt_vector字段权重为0.4

**4、融合得分（加权乘完之后，把两个字段的分值相加）**

该实体(final_score) = Σw * score

最后根据加权融合的得分从高到低排序，得到最终搜索结果的优先级。

#### 总结

Milvus规则：一个集合内的同一个向量字段，其度量类型必须统一。这是数据库设计的硬性规定。

WeightedRanker的应用：它用于处理一条实体有多个不同向量字段的情况，这些字段可以（也常常）使用不同的度量类型（IP， L2, Cosine）。

比较不同度量分数的方法：通过归一化技术，将所有路径返回的原始分数统一到相同的数值区间和方向上（通常为[0, 1]，且越大越好），然后再根据预设的权重进行加权求和，得到最终用于排序的综合分数。

### RRFRanker (Reciprocal Rank Fusion倒数排序融合)

**定义：**

一种融合多个独立排名列表的方法，目的是合并多个检索器（字段）返回的结果，使得最终结果能较好的反应所有检索器的优势，提高整体相关性和鲁棒性。

**一句话解释：**

final_score=字段1排名No.的倒数 + 字段n的排名No.倒数。分数越大排名越靠前。

RRF利用排名位置的倒数的累加来给每个文档打分。

**关键点：**

排名越靠前，倒数越大，贡献分数越高

k加上排名，有平滑作用，避免前几名分数差距过大

出现在多个排名列表且位置靠前的文档，雷击得分更高，最终排名优先。



### 混合向量检索

混合向量检索就是稀疏和密集向量检索，它俩的区别在哪里？

| 特性           | 稀疏向量检索 (Sparse Vector Retrieval)                       | 密集向量检索 (Dense Vector Retrieval)                        |
| -------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **核心思想**   | **精确匹配关键词**                                           | **理解语义和概念**                                           |
| **向量表示**   | 高维、稀疏（大部分元素为0）                                  | 相对低维、密集（所有元素都是浮点数）                         |
| **构建方式**   | 基于词频统计（如TF-IDF, BM25）                               | 基于深度学习模型（如BERT, Sentence-BERT）                    |
| **关键词处理** | **强依赖**关键词。查询和文档必须有共同词汇。                 | **不依赖**关键词。通过语义匹配，解决词汇鸿沟问题。           |
| **语义理解**   | 弱。无法理解同义词、缩写、上下位关系。                       | 强。能理解同义词、语义相关性、甚至隐含意图。                 |
| **典型代表**   | **BM25** (Bag of Words模型的巅峰)                            | **DPR, Sentence-BERT**等生成的Embedding                      |
| **优点**       | - 简单、高效、可解释性强 - 对精确术语匹配（如人名、产品号）效果好 | - 语义理解能力强，召回率高 - 对自然语言查询、口语化表达友好  |
| **缺点**       | - 无法解决词汇鸿沟问题（Vocabulary Gap） - 召回率可能较低    | - 需要大量数据训练模型 - 计算开销相对较大 - 可能错过精确术语匹配 - 可解释性差（“黑盒”） |
| **好比是**     | **图书馆的索引卡片系统**：你必须知道书名或作者名中的**确切词汇**才能找到书。 | **一个博学的图书管理员**：你可以用自己的话描述书的内容（“找一本关于太空冒险的搞笑小说”），他能理解并帮你找到。 |

### 总结

|              | **稀疏检索**                 | **密集检索**                     | **混合检索**                     |
| ------------ | ---------------------------- | -------------------------------- | -------------------------------- |
| **本质**     | 关键词匹配                   | 语义理解                         | **两者优势结合**                 |
| **目标**     | 保证**精度**                 | 提高**召回率**                   | **同时保证精度和召回率**         |
| **适用场景** | 精确查询、专家系统、法律文献 | 自然语言问答、推荐系统、长尾查询 | 现代搜索引擎、智能问答、知识库检 |

## 基于mysql数据库构建问答系统

### TF*IDF算法

（**T**erm **F**requency - **I**nverse **D**ocument **F**requency词频 - 逆文档频率）

**TF-IDF的核心思想是：**

一个词在一篇文章中的重要程度，正比于它在这篇文章中出现的次数，反比于它在所有文章中出现的次数。

- **在本篇出现越多，越重要**（TF）。
- **在别的篇出现越少，越独特**（IDF）。

- **既常见于本篇，又罕见于别篇的词，就是关键词**。
- **可以看到：越常见的词，IDF值越低（接近0）；越罕见、越独特的词，IDF值越高。**
- **最终计算：** 将TF和IDF相乘。
- **意义：** TF-IDF值的高低，代表一个词对一篇文档的代表性强弱。
- **高TF-IDF值：** 意味着一个词在当前文档中很常见（TF高），同时在所有文档中又很罕见（IDF高）。**这就是我们想要的关键词！**

### 总结与应用

**TF-IDF的原理**就是通过**TF**和**IDF**的相乘，巧妙地找出那些** locally frequent（局部频繁）** 但又 **globally rare（全局罕见）** 的词语，这些词语最能代表一篇文档的独特内容。

TF(词跟词的关系，以词数为单位计算)，

DF(文档和文档的关系，以文档数为单位)

IDF(等于DF的倒数)

IDF的智慧就在于它**颠覆了“常见即重要”的直觉**。它认为，真正重要的词，是那些能够帮助我们将**本篇文档**与**其他文档**区分开来的词。一个词在所有文档中都出现，它的区分能力就是0。



## 日志记录器

logging(记录库)，logging.getLogger('记录器1')

## BM25公式

BM25的核心思想是TF-IDF算法。最终评分是对单个查询词与文档的匹配计算。总得分是所有查询词得分的累加。

假设query是：手机发烫？

TF：“发烫”这个词出现在某个文档中概率很高，就代表匹配度很高

IDF：类似于权重，如果“发烫”这个词在总数1000条文档里都存在，就代表匹配度降低，

BM25 = TF分数 * IDF分数

BM25解决了TF-IDF文档长度不一致，和文档重复词出现的问题。并分别在此基础上对得分做出惩罚。主要是针对TF词频的算法进行改良。

RAG系统中，mysql系统检索，需要将query文档分词，做关键字检索。而milvus向量检索时，则不需要将向量进行分词，直接把文本字符串转成向量。做的是语义检索，而不是关键字检索。

## redis数据库

在redis数据库中存储数据，必须是键值对。也就是说key和value都必须是字符串类型，才能存，不支持字符串嵌套。{key = str, value = str}

所以client.set( key,json.dump(value) )；必须要把value转成字符串。

## MYSQL-QA总结

定义好mysql、redis的增删改查（set_data、get_answer、insert_data、fetch_answer）。检索时用户query先去bm25(query, threshold=0.85)做相似度匹配。

**step1：定义方法      _load_data()**

先把mysql中加载的所有的原始问题和分词后的问题存到redis当中。如果mysql中有更新answer的话，需要把redis的qa_original_questions和qa_tokenized_questions；清除，然后从mysql中从新加载一遍。从而给bm25(tokenized_question),备下比较文档tokenized_question。上述的两个qa_original_questions和qa_tokenized_questions只是用来备料而已，没有别的作用。redis中的高频问答对，是从mysql中检索出来后加进来的。

**step2：定义方法      search()**

先检查redis数据库中有没有添加过的问答对，没有才走mysql检索。有的话就return answer；return是方法的出口，执行完就代表方法执行结束，下面的检索mysql就不会执行了。

如果redis数据库中没有answer，就去mysql数据库中bm25检索，检索到最匹配的答案。连同query和answer，一并存入redis    方法：set_data。

相同问题再进来，直接进入redis中，进行匹配返回。







# 模块二、自问自答

## RAG是什么？

RAG（retrieve agument generation）检索增强生成。

检索：构建milvus、redis、mysql等数据库，以供查询。

增强：检索到和query相关的上下文，结合每一轮的query提示词，一直增加在history模块中，然后每次提问，都结合history数据，一起提供给大模型去生成。

生成：大模型作为大脑，结合问题和相关，思考生成回答。

## 为什么milvus数据库需要docker环境下安装，而mysql不需要。

不是MySQL“不需要”Docker，而是MySQL的传统部署方式已经做得太好，好到在生产环境中没有给Docker留下取代它的理由。

而像Milvus这样的新兴数据库，其传统部署方式异常复杂，反而让Docker成为了它的“标准”安装方式。

## milvus数据库中的索引，是什么意思？

这里的索引特指为了高效地进行向量相似性搜索而建立的一种数据结构。

目标是牺牲召回率，解决暴力搜索计算速度慢的问题，通过算法将向量空间进行划分、压缩和编码，建立索引。让检索系统，只检索最近的那个区域里的数据。相当于将图书馆的书，进行分门别类的整理。索引=类别=簇

方法：FLAT（暴力检索）、IVF系列（IVF_FLAT分簇、IVF_PQ量化、IVF_PQ）

相似度计算方法：L2欧式、IP点积，cos余弦相似度计算

## 多少条数据，才值得用ivf，而不是暴力搜索？

考虑的因素：

数据量：一万、十万、百万

向量维度：100、500、1000

性能要求：单个查询相应10ms以内，吞吐量：并发量。

准确率要求：

​	nprobe值越小，探查的簇就越少，搜索越快，召回率越低。可能错过真正的最近邻答案。

​	值越大，探查的簇越多，搜索越慢，召回率高，接近暴力搜索的结果。

## nprobe；这是词是哪几个单词组成的？

n代表 数量，probe代表 探查，合起来就是要检索的最相关的top-n 个探查点。例如，构建索引时，IVF会通过k means算法聚类成nlist个聚类中心。nlist就是列表的数量。

当query向量进来时，系统先计算它与所有nlist个聚类中心的距离，选出最近的nprobe个聚类中心，在这nprobe个聚类中心里面做暴力检索，计算query向量和该簇内所有向量之间的距离。

nprobe可以控制暴力检索的范围。nlist可以给数据分区。

nprobe=nlist；代表探查所有单元，此时IVF索引退化为暴力搜索。

nprobe（要探查的簇的数量）：是一个在搜索速度和结果精度（召回率）之间进行权衡的关键调优参数。

## IVF单词的组成是哪些？

IVF：inverted vector file （倒排向量文件）

比喻理解：

正排：通过书名找章节（关键字关键字是：深度学习，通过找所有的书，找到所有的章节）

倒排：通过章节找书名（关键字是：深度学习，找出所有包含“深度学习章节”的书）

## 聚类中心向量的计算公式

求所有向量的平均值；聚类中心向量=簇内所有向量相加，然后除以向量的个数。query先跟簇类中心向量做相似度计算。再和最相关簇类中心的所有向量进行相似度计算。

## k-means怎么聚类的？

先自设置n个中心点，然后每个向量距离这个中心点做一个欧式距离的距离计算，距离最近的就属于这个类，然后按照这个类里面的向量，重新计算簇心。

也就是先在数据样本中，随机定三个簇心，完成分簇。簇内再计算平均值作为新簇心。这个新簇心可能不是任何一个样本。

## 请帮我解释一下，docker容器中的milvus-redis、milvus-standalone、milvus-minio、milvus-redis、milvus-etcd，分别是什么意思，请解释一下？

这些容器共同组成了一个完整的Milvus standalone（单机版）部署。Milvus 作为一个复杂的向量数据库，其运行依赖于多个外部服务，因此官方使用Docker Compose将这些依赖服务一起启动，形成一个功能完备的系统。

**milvus-standalone**

含义：这是Milvus向量数据库最核心的组件，即Milvus服务器本身。

作用：它负责接收来自客户端（如Python/Java SDK）的请求，执行向量和标量数据的增、删、改、查（CRUD）等核心操作。你主要就是和这个服务进行交互。

milvus-standalone 这个容器名称正是指 Milvus 的单机版部署模式。

**milvus-etcd**

含义：这是一个分布式键值存储系统。Etcd是Milvus的元数据管理引擎。

作用：它负责存储和管理Milvus的所有元数据（Meta Data）。

元数据包括：集合（Collection）和分区（Partition）的结构信息（如名称、维度、索引类型）	数据段（Segment）的信息。	消息的时间戳定位信息。集群节点的健康状态和负载信息。	

重要性：它保证了元数据的一致性和高可用性。如果etcd服务不可用，Milvus将无法知晓数据的结构，从而导致系统瘫痪。

类比：相当于网站的服务发现和配置中心，或者是一个非常可靠且快速的系统“户口本”和“目录册”。

**milvus-minio**

“MinIO” 这个名字是其核心功能的完美体现，它是由 Mini 和 IO 两个部分组成的合成词。

Mini： 意为“迷你的”、“极小的”。这指的是该软件轻量级的特性。与传统的、庞大的企业级存储解决方案相比，MinIO 的二进制文件非常小，资源占用极低，部署和运行都非常简单快捷。

IO： 这是计算机领域的通用缩写，代表 Input/Output（输入/输出）。这直接指明了它的核心功能——数据的存储（输入）和检索（输出）。

因此，MinIO 的字面意思和设计哲学就是“轻量级的输入输出（存储系统）”。

你可以把它理解为：

一个极其轻便、高效的数据存取工具。

它的名字就告诉了你它的两大优点：小巧（Mini） 和 高速（IO，高性能读写）。

**milvus-redis**

含义：这是一个非常流行的内存键值数据库。在较新版本的Milvus（如你这里的2.4.4）中，Redis的作用已经发生了变化。

类比：相当于系统内部的消息总线或任务调度中心，确保指令高效、准确地传递。

## 那么请问milvus数据库中，是把一个字符串文档怎么存成稀疏向量的？假设词表为2万个。意味着每一个稀疏向量都是2万维吗？然后把具备该文档词汇的对应位置设置成BM25分数？没有的就是0？不太可能吧。

### 存入稀疏向量的流程

**1、文档存入milvus之前的预处理：**

构建词表（已有的）

对每个文档进行分词

为每个文档计算每个词的TF-IDF权重

将每个文档转换为（indexes，values）的格式。也就是（词表中所在编号，该词在该文档中对应的BM25分数值

**2、定义集合collection**

在milvus中定义一个集合，其中schema包含一个稀疏向量字段，milvus会自动识别这种元组格式。

**3、插入数据**

将上面准备好的（indexes，values）插入到milvus中。

**4、检索：**

当进行检索时，查询语句Query（e.g. “我 爱 你”）也会被转为同样格式的稀疏向量（计算每个词的IDF座位权重）。query和doc格式要一致。

查询向量：index_q = [0,1,2]，values_q = [IDF(“我”), IDF(“爱”), IDF(“你”)]

Milvus会高效地计算查询向量与所有文档向量的点积（实际上是计算它们共享词汇的权重乘积之和），并返回分数最高的文档。

## redis的数据存储格式是什么样的？

高频问答对：

{ “用户问题的哈希值转化”：{ 答案：“ ”，源文档：“ ” } }

"cache:f6d5e0e..."` : `"{\"answer\": \"牛顿第一定律...\", \"source_doc\": \"...\"}"

## redis匹配机制是什么？要完全和用户的query完全一致吗？

分三种情况：精确匹配、归一化匹配、语义匹配

**1、精确匹配（简单的哈希策略）**：用户query的哈希转化值 对比 查询redis存储的问题的哈希值。一个标点符号都不能差，否则就检索不到，进入RAG系统检索。

**2、归一化匹配：**生成hash值前，先对用户的原始query（无论是正要查询的，还是已经存进去的query）都做清洗和归一化处理：

**去除首尾空格、转换为全小写，去除所有标点符号，其他语言转成简体。**

**3、语义缓存（高级，更智能）：**key不再基于问题的字面本身，而是基于问题的语义。

**工作原理如下：**

1、将正在查询的query用embedding模型转成向量

2、在redis的“query向量”库，做相似度计算。搜索相似度超过阈值例如0.85的缓存记录。找到就返回。

优点是：能处理词不一样，但是语义相似的，提高召回率。

缺点：计算相似度，需要额外开销，并且需要维护一个向量的相似度搜索。

## 在RAG系统中，redis数据通常是高频问答对。请问redis通常有几个字段？将query转成hash值的query字段算一个，那么answer字段呢？是字符串吗还是字典类型？

**总结**

1. **Redis中存储的是字符串**：无论是键值对的key还是value，最终在Redis中都只能存储为字符串
2. **Answer字段通常是字符串**：可以是纯文本或JSON格式下的字符串形式字典
3. **推荐使用JSON序列化**：可以把字典、列表、元组等python复杂数据类型转化为client.set(key, value)可识别的json格式的字符串类型或者转成redis的数据类型：比如python的字典=redis的Hash，python的列表=redis的List，set=Set。但是它们通通可以转为redis可识别的json格式的字符串。
4. **Query哈希作为key**：确保唯一性和快速查找
5. **包含元数据字段**：便于缓存管理和质量评估

**redis数据库不存在表这一说。相当于mysql中的一个数据库，一张表。表中有两个字段，一个是hash(query)字段，一个是answer字符串字段。不过是json格式的，表现为字典字样。查询时load还原成字典，即可返回查询。**

**操作有：**

增（client.set()）

删( client.delete() )

查( client.get() )、（ client.keys()）

## 终端运行也能成功，导包必须注意

在当前要运行的py文件中，告诉你要导包所在的路径即可（也就是把你要导的包的路径加到sys系统路径变量里）。

相当于安装一个软件，然后给他配一个系统环境path变量

```python3
integrated_qa_system\mysql_qa\db\mysql_client.py

import os
import sys

# 找到根路径path3
path_0 = os.path.abspath(__file__)
path_1 = os.path.dirname(path_0)
path_2 = os.path.dirname(path_1)
path_3 = os.path.dirname(path_2)

# 把根目录名和config、logger两个py文件所在的文件夹路径名base拼接起来
base_path = os.path.join(path_3,"base")
# 得到base_path,插入到sys.path里面，让mysql_client.py找的到config和logger
sys.path.insert(0, base_path)

# 导包
from config import Config
from logger import logger
```



# 模块三 milvus检索

## milvus检索，整体架构与工程流程

### 代码目录结构

prompts.py		提示词模版

query_classifier.py		查询分类器

strategy_selector.py		检索策略选择器

vector_store.py		向量储存与检索

rag_system.py 		rag系统核心逻辑

### milvus系统基本工作流程

#### note1：查询分类

意图识别（调用二分类模型，区分是通用知识还是专业查询？通用知识直接调用大模型回答，专业知识就需要经过Milvus检索。因为milvus检索非常慢，是最后一步）；

e.g.通用知识：华为手机在哪个平台购买，最可靠？；领域知识：我货号为xxx的手机现在到哪里了？

![1757755008895](C:\Users\gan\AppData\Roaming\Typora\typora-user-images\1757755008895.png)

#### note2：选择检索策略”、“文档检索”

优化部分，一开始是直接进入milvus系统进行检索，但后续项目迭代优化，做了“选择检索策略”、“文档检索”等一系列优化。

主要原因是用户的query，比较抽象和复杂，导致检索效果不佳

**选择检索策略：**

根据查询特点选择检索策略：本质上是为了更精确的检索到query相似度最高的上下文。

​	直接检索：适用于明确查询

​	HyDE检索：适用于抽象问题，生成假设答案后检索。

实现流程：先将query送入大模型，大模型理解后得到一个答案，基于这个答案再去检索，检索到的上下文和用户的query做一个拼接。一并送入大模型做理解式生成最终答案。

​	子查询检索：分解复杂查询。主要场景是，用户query中涉及多个实体，需要拆解成多个子问题进行查询，比如mate70对比mate60，有哪些优化？需要拆分成mate70的功能点是什么？mate70的功能点有什么？

​	回溯检索：简化复杂问题后检索

## 项目迭代

### 	V1: 基础的RAG系统（无任何优化）

实现流程：

**加载与切分文档**：从多种类文档（txt、pdf）加载并切分成块

**向量存储**：embedding模型转化文档为稠密向量储存

**答案生成：**基于检索到的上下文送给大模型进行生成回答

###		V2: 对查询做一个增强（检索策略），增强检索精度

实现流程：

**假设文档嵌入：**当用户的query检索不到数据时，就用大模型针对用户query做一个假设性回答，基于这个回答，再去检索答案文档。最后把检索到的文档+用户query一并送给大模型生成答案返回。

**子查询：**当用户的查询涉及多个实体时，分解成多个实体去查，也就是分成多个query。

**回溯提示**：把用户的query做一个意图提炼和查询重写，就是把用户的query的中心思想用另一句话表达出来。

### 		V3: 增强索引 （加快检索速度）

实现流程：

**多粒度切块：** 存数据时，是一块一块的存，现在把 “块” 分成子块和父块两种粒度：子块和对应的父块。检索时，先搜索子块，如果检索出的topk个子块中，有特定n个子块属于同一个父块，就把这个父块作为上下文信息，提供给LLM。

这是因为用户的query往往比较短，小的可能只有4个token。但是我们切块一般是按照200到500 个token的数量去切，如果用4个token去和500个token的文档做相似度计算，效果会很差。但是如果把500个token的父块，拆成10(超参数)个子块，也就是50个token。相似度计算精度就会提升很多。

**混合检索：**把文档转成向量时，有两种方法，一个是embedding模型转化的dense vector，稠密向量。一个是关键字算法bm25，得到的每一个关键字的bm25分数，这是稀疏向量（query中的分词都做bm25计算，子块向量也根据query的分词，做bm25计算）。最后用重排RRF对稀疏和稠密检索到的上下文进行一个排序，让大模型根据顺序进行权重分配，生成最终的答案。

### 		V4: 改进检索器

核心功能：

**元数据过滤：**比如milvus数据库做增删改查操作的时候，可以加一个filter字段，也就是条件查询，比如设置schema时的动态字段：{"color":"pink_8682"}。filter = color=pink。这可以减少不必要的检索。相当于按照部门来区分，比如功耗类，性能类

**查询场景：**我们是海思下面的kirin产品线，专做kirin知识检索。专门给用户一个提示词，把用户的问题从源头做一个分类：

按按研发阶段分类

按知识领域分类、

按知识类型分类

### **Kirin产品线RAG系统知识分类体系**

我们将采用 **“研发阶段 + 知识领域 + 知识类型”** 的三维过滤体系，这是最符合芯片开发流程的方式。

#### **第一级：按研发阶段分类 (Development Phase)**

这是芯片开发的核心脉络，工程师通常很清楚自己当前所处的阶段。**（这是最高效的一级过滤器）**

| 分类标签                           | 说明                                                         | 典型查询示例                            |
| ---------------------------------- | ------------------------------------------------------------ | --------------------------------------- |
| **architecture** (架构设计)        | 芯片spec定义、IP选型、性能功耗面积（PPA）目标设定、系统框图。 | “Kirin 9000的CPU集群架构是怎样的？”     |
| **frontend_design** (前端设计)     | RTL编码、IP集成、低功耗设计（UPF）、功能仿真。               | “如何为GPU模块编写UPF文件？”            |
| **verification** (功能验证)        | 模块级/芯片级验证、UVM、形式验证、功耗感知验证、代码覆盖率。 | “如何为DPU模块生成随机测试向量？”       |
| **physical_design** (物理实现)     | 逻辑综合、布局布线、时钟树综合、时序签核、物理签核（DRC/LVS）。 | “Innovus中修复建立时间违例的常用命令？” |
| **analog_rf_design** (模拟/RF设计) | 特定于Kirin的PMIC、音频编解码器、射频收发器等模拟IP设计。    | “LPDDR5 PHY的阻抗校准流程？”            |
| **sdk_development** (SDK开发)      | 芯片底层驱动、Bootloader、固件、Hal层开发。                  | “如何配置NPU的中断寄存器？”             |
| **post_silicon** (硅后测试)        | **（你的主场）** CP/FT测试、系统级功能/性能/功耗/可靠性测试、故障诊断。 | “游戏场景下GPU功耗超标的排查步骤？”     |
| **customer_support** (客户支持)    | 面向手机厂商（如华为终端）的参考设计、问题排查、调试支持。   | “某型号手机摄像头启动慢如何定位？”      |

#### **第二级：按知识领域分类 (Knowledge Domain)**

这是在“研发阶段”下的二次细分，指向具体的技术模块。

| 分类标签                         | 说明                                     | 通常从属于哪个阶段                              |
| -------------------------------- | ---------------------------------------- | ----------------------------------------------- |
| **cpu** (CPU)                    | ARM核、微架构、调度                      | architecture, verification, post_silicon        |
| **gpu** (GPU)                    | Mali/自研GPU、图形驱动                   | architecture, verification, sdk_development     |
| **npu** (NPU)                    | AI处理器、算子库                         | architecture, sdk_development, post_silicon     |
| **isp** (ISP)                    | 图像信号处理器、图像调优                 | verification, sdk_development, customer_support |
| **modem** (通信基带)             | 5G/4G协议、射频校准                      | analog_rf_design, post_silicon                  |
| **mm** (多媒体)                  | 视频编解码（VDE）、显示处理（DPU）       | verification, sdk_development                   |
| **connectivity** (连接性)        | Wi-Fi 7、蓝牙、GPS                       | analog_rf_design, post_silicon                  |
| **power_thermal** (功耗与热管理) | **你的核心领域**：DVFS、电源域、温控策略 | architecture, physical_design, **post_silicon** |
| **memory** (存储子系统)          | LPDDR5、UFS、缓存一致性                  | physical_design, verification                   |
| **security** (安全)              | TrustZone、加密引擎、安全启动            | architecture, sdk_development                   |
| **dfx** (可测试性设计)           | DFT、DFD（可调试性设计）                 | frontend_design, post_silicon                   |

#### **第三级：按知识类型分类 (Knowledge Type)**

定义知识的形态和权威性，用于筛选信息的“原材料”。

| 分类标签                         | 说明                                                     |
| -------------------------------- | -------------------------------------------------------- |
| **design_spec** (设计规范)       | 最权威的需求文档，如《Kirin XYZ CPU子系统设计规范》      |
| **test_spec** (测试规范)         | 《Kirin XYZ 硅后功耗测试规范》                           |
| **verification_plan** (验证计划) | 《ISP模块验证计划》                                      |
| **rca_report** (问题案例)        | **最宝贵！** 《20240501_某游戏闪退_GPU功耗毛刺_RCA报告》 |
| **application_note** (应用笔记)  | 《如何调试NPU模型精度损失问题》                          |
| **tool_manual** (工具手册)       | 内部功耗分析平台、示波器、ATE机台的使用指南              |
| **review_material** (评审材料)   | TR4/TR5评审会议纪要，记录关键决策和风险点                |
| **standard** (行业标准)          | 3GPP、JEDEC、MIPI等标准协议                              |

------

###		V5 自我反思

用大模型评估我检索出的结果和query的相关性，如果达不到我要的指标，就让RAG再次进行检索。

### 		V6路由选择：为了提升时间效率

做了一个二分类，意图识别。考虑到通用知识也进入RAG检索的话，就时间效率太低了。

# RAG系统评估

## 定义

全称是：retrieval augmented generation assessment。即检索增强生成的自动评估。评估主要基于两个方面：检索retrieval和生成generation。生成好坏决定是否要更换模型，更换成参数量更大的模型，检索效果好坏决定是否要改retrieval相关代码和超参数。例如切块的大小等等

评估所需数据集：

query：作为RAG管道输入的用户查询，输入。

answer：从RAG管道生成的答案，输出。（predict）rag系统获取，写到数据集里

contexts：从用于回答question外部知识源中检索的上下文。rag系统获取，写到数据集里

ground_truths：question的基本事实答案，这是唯一人工注释的信息。（label）



### 指标：

**1、answer relevancy：**（这属于**“生成”**生成效果的评估，锚点是大模型的**“回答”**）

代表**query**和**answer**之间的关系。

```
原理：
基于生成的答案；用大模型对其生成n个question。再用余弦相似度计算question和用户query的相似度。分数越高，相关性越好。

实现方式：
为了估计答案的相关性，提示大模型根据给定的答案生成N个潜在问题Qi；编写如下提示词：
为给定答案生成一个问题
答案：[答案]

具体步骤：（生成一个question足以）
1、对给定答案，提示大模型(LLM)生成基于该答案可能得n个问题Qi
2、使用embedding模型获取所有问题的向量。
3、对question，计算它与用户query之间的相似度，也就是query向量和question向量的余弦相似度计算
4、答案相关性分数=N个 余弦计算(question1，query1) 之和；最后除以N
```

**2、faithfulness：**（这属于**“生成”**生成效果的评估，锚点是大模型的**“回答”**）

代表**answer**和**context**之间的关系

```
定义：
忠实度是指答案是根据给定的上下文得到，这对于避免错觉，并确保检索到的上下文可以用作生成答案的依据非常重要。
如果分数低，表明大模型的回应没有遵循检索到的知识，提供幻觉式答案的可能性增加。
实现方式：
提示词1：
给定一个问题和答案，从给定答案的每个句子中创建一个或多个陈述。
问题：[问题]
答案：[答案]

生成从答案中推断出的陈述后，LLM 判断每个陈述 Si 是否可以从 上下文 中推断出来。这个验证步骤使用以下提示进行：
提示词2：
考虑给定的上下文和以下陈述，然后确定它们是否由上下文中的信息支持。在得出结论（是/否）之前，为每个陈述提供简要解释。在最后以给定格式为每个陈述提供最终结论。不要偏离指定的格式。
陈述：[陈述 1]
...
陈述：[陈述 n]

计算公式：F=V / S；其中V等于大模型支持的陈述数量， S表示陈述的总数。
注释：大模型基于生成的答案的每一个句子，生成一段描述的句子，共有S个句子。这些句子可以根据上下文推断出来的有V=(S-M)个。
```

**3、context precision：**（这属于**“检索”**效果的评估，中心点是你检索出来的**“上下文”**）

代表**query**和**context**之间的关系

```
定义：
context precision也就是context应该只包含回答问题所需的信息，惩罚包含冗余信息的情况。
也就是说检索到的上下文中的废话越多，就代表，检索效果越差，把无用的信息检索出来了。
比率越高，表示检索到的上下文与问题的相关性越强。

实现方式：
为了估算 上下文 和大模型的 回答 的相关性，可以借助提示词的形式让大模型帮我们找出来，上下文中抽取出对回答问题query至关重要的句子S

prompt：
请从提供的{上下文}中提取可能有助于回答以下{问题}的相关句子，
如果没有找到想相关句子，或者你认为问题无法从给定上下文中得到回答，
则返回短语“信息不足”
在提取候选句子时，你不得更改给定上下文中的句子

计算公式 = 上下文中提取的有助于回答query的句子数量num \ 上下文中的句子总数
e.g. 上下文中提取的有助于回答query的句子数为 5；上下文中句子数一共有10句
则上下文精确率=5\10=0.5
```

**4、context recall：**（这属于**“检索”**效果的评估，中心点是你检索出来的**“上下文”**）

代表**context**和**ground truths**之间的关系

```
原理：
衡量检索到的上下文context和真实答案ground truths的匹配程度
该指标通过 标注答案和检索到的上下文计算，分数范围在0到1之间，得分越高表示性能越好。

要从真实答案中估计上下文召回率，需要分析真实答案中的每个句子，以确定它是否可以归因于检索到的上下文。如果召回率为100%，则真实答案中每个句子，大模型都能在检索到的上下文中，找到相同的句子。

计算公式：context recall = （真实答案句子=上下文句子）的句子个数 \ 真实答案的句子数

分子：表示在真实答案（GT）中的论断中，有多少是可以归因于检索到的上下文的。换句话说，这些论断在检索到的上下文中找到了支持或者依据。也就是说：哪些真实答案的句子是可以在上下文中被检索到的
分母：表示真实答案中论断的总数量，也就是句子的数量。

```

### 总结：

faithfulness忠实度：答案是否忠于上下文。答案的句子能不能在上下文中找到

answer relevancy：基于答案生成question。question和query是否相似

context precision：上下文的句子中，有多少句子，对回答用户query是有帮助的？

context recall：人工写的答案，在上下文中找到多少？

### 最重要的两个指标：以生成为优先级最高的依据

faithfulness：值越高，说明答案幻觉越少

answer relevancy：值越高，说明答案和用户query的相关性越强。（这个是最重要的）

还可以人工去评估。给rag一个query，生成一个答案。人为评估，结果怎么样。

RAG AS+人工评估

也就是faithfulness + answer relevancy + 人工评估

为避免调用本地模型api调用超时，可以买评估模型的api key。

### 项目版本迭代的原因：

#### answer relevancy   代表**query**和**answer**之间的关系

问题原因：如果答案相关性低，说明大模型生成的答案偏离了用户意图。

1、提示词工程：在提示词中明确要求模型“严格基于用户问题回答”，避免冗余或泛化。

2、查询处理：对用户查询进行重写（query rewriting）、扩展（query expansion）或意图识别（intent detection），以更精准地表达用户需求。

#### faithfulness：代表**answer**和**context**之间的关系

如果忠诚度低，说明模型产生了“幻觉”（hallucination），即生成了context中不存在的内容。

提示词约束：在提示词中强调“仅基于提供的上下文回答”，并设置安全回复（如“未在知识库中查询到相关信息”）。

检索增强：确保检索到的context质量高（如提高检索精度），减少模型依赖自身知识的可能性。

####  context precision：代表query和context之间的关系

上下文精确率低，代表RAG检索效率低。加入细粒度划分，也就是切分子父块。以及混合检索模块，加强对query的精细化操作

1、使用混合检索（结合关键词检索与向量检索）提升查全率和查准率。

2、对文档进行细粒度分块（如父-子块结构），并优化块大小和重叠度。

3、添加元数据过滤（按来源、日期等筛选）。

4、对查询进行嵌入优化（embedding optimization）或使用RAG-Fusion等重排序技术。

#### context recall：代表context和ground truths之间的关系。

**问题分析**：检索系统未能充分召回知识库中的相关信息。

**优化方案**：

1、**扩大检索范围**（如增加top-k检索数量）

2、**智能检索策略**

- **子查询分解**：针对复杂长查询，拆解为多个子查询并行处理
- **假设问题生成**：为模糊查询自动生成多个假设性问题
- **直接查询优化**：完善简单问题的快速检索通道
- **回溯检索**：针对大数据量查询建立回溯检索机制

#### **响应速度慢，就改变索引和分类查询模块**

1、缓存机制：对常见查询及答案进行缓存。

2、并行处理：并行执行检索与生成步骤（如异步调用）。

3、模型轻量化：使用蒸馏模型、量化或更小的生成模型。

# 基于milvus构建RAG系统

## 文档处理模块    process_documents

定义document_loaders()，里面元素是字典格式{key=后缀，value=加载器名称}

### OCR

```
def get_ocr(use_cuda: bool = True) -> "RapidOCR":
    try:
        from rapidocr_paddle import RapidOCR
        '''
        1、det_use_cuda=True：启用检测模型的GPU加速。
        text detection文字检测，帮助找出图片中有文字的区域，也就是包围盒。
        2、cls_use_cuda=True：启用分类模型的GPU加速。classification分图片方向
        text classification，在完成文字识别前，帮助判断文字的方向（如0°，90°，180度，270°）
        3、rec_use_cuda=True：启用识别模型的GPU加速。recognition 识别图片
        text recognition 帮助将检测出的文字区域图像转换成文本。
        '''
        # 如果不显式，写出这些参数，默认使用CPU计算。
        ocr = RapidOCR(det_use_cuda=use_cuda, cls_use_cuda=use_cuda, rec_use_cuda=use_cuda)
    except ImportError:
        from rapidocr_onnxruntime import RapidOCR
        ocr = RapidOCR()
    return ocr
```

paddleocr：作用是解析图片中的文字，也可以进行表格识别

rapidocr_paddle和rapidocr_onnxruntime 两种导入方式

主要区别在于它们所使用的推理引擎和硬件支持

选择哪种方式最合适取决于你的硬件环境和性能需求

当你有GPU且追求速度时：使用rapidocr_paddle，PaddlePaddle原生就支持在GPU上推理PaddleOCR模型，速度更快。

当只有CPU且需要高效推理时：使用rapidocr_onnxruntime( Open Neural Network Exchange 开放神经网络交换)。它在CPU上进行了优化，资源占用较低。

### docloader.py

```
# 注意：命名疑惑：import fitz 但安装 pymupdf（pip install pyMuPDF）。fitz是PyMuPDF的接口模块
import fitz # PyMuPDF所依赖的底层C库--MuPDF，其开发者Tor Andersson，Fitz是他的昵称之一。因此得名。
```

作用：fitz（pyMuPDF）是一个功能极其强大且速度飞快的python库，主要用于处理PDF文件。

适用场景如下：它尤其适合需要批量处理大量PDF文件或对性能要求极高的应用场景。

1. 高精度、高效率的从PDF中提取文本和图片
2. 将pdf页面转换为图像
3. 对PDF进行合并、拆分、旋转、加密等操作    

支持对PDF等文档进行读取、编辑、创建、注释、提取内容等操作。

**例如文本提取：**从PDF中提取文字内容，支持按页、按块、按行等粒度去读

```
import fitz
doc = fitz.open("example.pdf")
for page in doc:
	text = page.get_text()
	print(text)
```

### DIL（python imaging library）

定义：一个强大的，用于图像处理的python第三方库，原始PIL，于2009年停止维护。后续的PIL都是分支：Pillow

PIL/Pillow 是PDF文字识别流程中不可或缺的“前置工序”，通过优化图像质量来显著提高最终文字识别的准确率。它的作用是将对图像进行灰度化、二值化、降噪、增强等操作，为OCR准备最佳输入。

### cv2（import cv2）

定义 ：cv2是OpenCV（open source computer vision library）的python接口导入名，它是一个功能极其强大的开源计算机视觉和机器学习软件库。

在rag系统中，cv2主要作用是作为PIL/Pillow的一个补充，提供更高级和多样的图像预处理功能。























## query意图识别

### 导包

**1、从transformer库导入  分词器和分类方法**

from transformers import BertTokenizer, BertForSequenceClassification

**2、训练库trainer和导入训练参数的方法TrainingArguments**

from transformers import Trainer, TrainingArguments

**3、训练和验证集的划分**

from sklearn.model_selection import train_test_split

**4、评估分数，主要以F1值为主**

from sklearn.metrics import classification_report, confusion_matrix

### trainer和training arguments的使用



## 初始化方法

### 初始化milvus客户端的，一系列参数，包括collection name、host、port、database等；以及日志记录器、BGE-Reranker模型、BGE-M3这个稠密向量的embedding函数。

### def _create_or_load_collection(self) 创建或加载milvus集合，定义字段、schema和索引参数。为稠密和稀疏向量添加索引。

### add_documents(self, documents)；BGE-M3的embedding模型，为文档转化出稠密向量，

## 文档解析工具

背景：为了从搜集到的文档中提取到知识信息给RAG增加知识库来源，系统实现了一系列专门的加载器（loaders），这些加载器可以提取文档中的文本信息，还可以利用OCR技术识别并提取图片中嵌入的文字。它们都继承自Langchain的BaseLoader，并实现了lazy_load方法来按需生成Document对象。

### langchain框架的使用：

langchain.com（中文：docs.langchain.com.cn）主页里的Doc目录下有pathon编写的langchain库，里面有丰富的langchain的API接口方法，其中有langchain的，langchain-community，langchain-core的，同时还有其他大模型厂商提供的一系列的接口方法。



# 融合Mysql的RAG系统

## 系统API接口开发和web UI应用

### FastAPI部署的作用

FastAPI 的强大之处就在于，它同时完美支持了构建传统的 RESTful HTTP API **和** 现代的 WebSocket 功能，让你在一个框架内就能解决大多数前后端通信的需求。

### 在py程序里面启动FastAPI

```
# main.py

from fastapi import FastAPI，WebSocket
import os

# 创建 FastAPI 应用实例，
# 设置标题和描述
app = FastAPI(title="问答系统API", description="集成MySQL和RAG的智能问答系统")

@app.get("/") # 函数装饰器（有调用、有返回、有增强、有嵌套），一般用于home界面的接口。
async def root():
	import uvicorn
	return {"message":"Hello World"}
		
if __name__ == "__main__":
    import uvicorn
    # 可以从环境变量获取端口，如果获取不到则使用默认值 8000
    port = int(os.getenv("PORT", 8000))
    # 以编程方式启动，参数是字符串 "main:app"，# 运行 FastAPI 应用，监听 0.0.0.0:8000
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
```

### 在终端cmd中启动FastAPI（推荐）

```
解释：
unicorn是用来启动FastAPI的；FastAPI启动之后，表示它修饰的函数作为一个应用，被分配了默认的端口号8000；也就是说网页上输入：server ip：8000就可以定位到这个函数，在客户端发出请求，app.get()监听到来自客户端的请求，就会把函数的结果返回给客户端，继而得以显示。

注意：（先进入conda指定环境，因为unicorn被install在里面）

e.g. 核心主程序是main.py ；启动命令就是：
unicorn {入口文件名} ：{FastAPI()的实例化对象名} --reload
unicorn main:app --host 0.0.0.0  --port 8000 --reload

开发阶段：强烈推荐使用命令行方式	uvicorn main:app --reload
因为 --reload 参数极其方便，你不需要每次修改代码后都手动重启服务器。
```

### 实现FastAPI，函数接口的简单调用的代码如下：

```
from fastapi import FastAPI，WebSocket
# 创建 FastAPI 应用实例，
# 设置标题和描述
app = FastAPI(title="问答系统API", description="集成MySQL和RAG的智能问答系统")
@app.get("/") # 函数装饰器（有调用、有返回、有增强、有嵌套）
async def root():
	return {"message":"Hello World"}
```



### 代码解释

**1、from fastapi import FastAPI，WebSocket**

```

from fastapi import FastAPI #  是导入创建主应用和定义普通 HTTP API 接口的类。
from fastapi import WebSocket # 是导入专门用于处理 WebSocket 连接的类，让你能够创建那些需要实时、双向通信功能的接口。

app = FastAPI() 实例化这个库
```

总结对比：

| 特性         | HTTP (FastAPI 常规路由)           | WebSocket                       |
| ------------ | --------------------------------- | ------------------------------- |
| **通信模式** | **请求-响应** (客户端主导)        | **全双工双向通信** (双方平等)   |
| **连接状态** | **无状态** (每次请求后连接可关闭) | **持久化** (一旦建立，长期保持) |
| **数据流向** | 单向 (客户端请求 -> 服务器响应)   | 双向 (客户端或服务器可随时发送) |
| **实时性**   | 较低 (依赖客户端轮询)             | **非常高** (服务器可主动推送)   |
| **开销**     | 每次请求都有HTTP头信息，较大      | 建立连接后开销很小，只有数据帧  |

2、@app.get("/") 

```
@，这个符号是装饰器的格式；
app是FastAPI的实例化对象
get是http协议的监听方法，监听客户端的GET请求，返回所装饰的函数的运行后的到的结果
("/") 这是路由；由路寻函。写/，代表根路径，至少要写一个路径名。目的是为了区别调用的函数。
e.g.https://www.ai-engineer.top/search/
https://www.ai-engineer.top；这个相当于目标服务器所在地址，
/search/相当于@app.get("/search")；专门顺着这个路径找到 搜索 这个函数，并返回执行结果。
https://www.ai-engineer.top/about；就是点击“关于”，然后返回这个函数的信息给请求端。

类比：
https://129.80.32.33：8000/search？什么是快乐星球：就相当于你在129.80.32.33这台电脑上，打开了一个应用（名叫FastAPI）,你在这个应用里面点击搜索，并输入查询了“什么是快乐星球？”。
而客户端监听到你的请求，会去找search这个函数，把根据query检索到答案给你传回去。由此完成了一次简单的http协议的post(上传)和get请求。完成了数据在客户端和服务器端的数据交互和传递。



那些应用里的功能，其实就是一个一个函数。上述只是以及目录。二级目录则相当于：银行app的-账户余额-消费记录；就相当于@app.get("/home/Account_left/records/")

```

大模型纠正：

```
这个比喻可以更精确一点：
FastAPI 应用本身不提供前端界面（那些按钮和输入框）。它只是一个提供数据接口（API） 的后端程序。

更准确的比喻是：
你打开一个浏览器（或其他客户端），在地址栏里手动输入了 https://129.80.32.33:8000/search?q=什么是快乐星球 并回车。

你的浏览器（客户端）向服务器发起了一个 GET 请求。

服务器上的 FastAPI “接线员”看到路径是 /home，就把这个请求转给了 search 函数。

search 函数拿到参数 什么是快乐星球，执行检索逻辑，然后把结果数据（通常是JSON格式） 返回给你的浏览器。

你的浏览器再以某种形式（可能是另一个前端页面）把这些数据展示给你看。

为什么这个区别重要？ 因为 FastAPI 的强大之处在于它是“前后端分离”的典范。它的后端只负责处理数据和逻辑，而“点击”按钮这个动作和漂亮的页面，是由另一个前端程序（如 React, Vue.js 开发的网页）来负责的。
```

大模型精修答案：

```
“@app.get("/") 中，
@是装饰器，
app是FastAPI实例。
get用于监听GET请求，并将返回值发给客户端
。("/")是路由，其核心原理是‘由路寻函’，
通过不同的路径（如/search、/about）来区分并调用不同的函数。

例如，访问 https://www.ai-engineer.top/search?q=什么是快乐星球 时，
www.ai-engineer.top 找到了服务器，
路径 /search 让服务器上的FastAPI应用找到了对应的搜索函数，
查询参数 q=什么是快乐星球 作为输入传递给函数，函数执行后的结果再通过HTTP GET 请求的响应返回给客户端，完成了一次完整的数据交互。”
```

注释：

当配置了app最基础的部分后，可以通过访问服务器根路由"/"，来得到函数返回的东西，也就是{"message":"Hello World"}；这是FastAPI开发的最简单的基于http协议的后端服务（server）应用。

### async异步和sync同步的区别

| 特性         | 异步 (`async def`)                            | 同步 (`def`)                        |
| ------------ | --------------------------------------------- | ----------------------------------- |
| **核心思想** | 非阻塞，等待时让出资源                        | 阻塞，等待时占用资源                |
| **关键字**   | `async`, `await`                              | 无                                  |
| **适用场景** | **I/O 密集型** (数据库、网络请求)             | **CPU 密集型** (复杂计算、数据处理) |
| **性能**     | 高并发、高吞吐量                              | 受限于线程池大小                    |
| **库依赖**   | 必须使用支持异步的库（如 `asyncpg`, `httpx`） | 可以使用任何普通的同步库            |

### 接口个数

```
# 1、创建新会话接口
@app.post("/api/create_session")

# 2、查询历史消息接口	# 返回会话 ID 和历史记录  #{session_id}这是动态路由
@app.get("/api/history/{session_id}")

# 3、清除历史消息接口	# 根据对话id 清除指定会话的历史记录
@app.delete("/api/history/{session_id}")

# 4、非流式查询接口		# 返回 JSON 格式响应答案和，记录处理时间。
@app.post("/api/query")

# 5、流式查询WebSocket接口	 #发送start、token、end或error类型的 JSON 消息，异步处理确保实时性
@app.websocket("/api/stream")

# 6、健康检查接口	# 返回健康状态
@app.get("/health")

# 7、获取问题类别接口	# 返回系统支持的问题类别。，便于前端动态展示过滤选项。
@app.get("/api/sources")
https://www.zhipin.com/job_detail/09538e53edbfb5dc03N52N2_FFVU.html?securityId=FBEIjlGtLLxzu-K195XTu_XEFEsH__wq-xpkwK_xtfrYzMT7HiP1ENycB1cahFV8ie8sEWpyjGO_zyyQj1ivYF7OL-DqAHOPH9TST6e08d5BYnw49nkh6yMiNQ~~
```

### 前端是怎么接收后端函数返回的json数据的？

```
按下F12，进入dev tool开发工具页面，里面有sources，下面有很多js文件，里面可能定义一些方法：
比如：
// 这是一个异步函数，用于获取用户数据
async function getUserInfo(userId) {
  try {
    // 1. 前端使用 fetch 向后端API地址发送GET请求
    const response = await fetch(`https://www.zhipin.com/beijing/web/common/getUserInfo?id=${userId}`);

    // 2. 解析后端返回的JSON数据
    const userData = await response.json();

    // 3. 将数据更新到网页上
    console.log('获取到的用户数据:', userData);
     // 假设这是一个更新页面的函数
    updateUserProfileOnPage(userData);

  } catch (error) {
    // 如果请求失败（如网络错误、API报错等）
    console.error('获取用户信息失败:', error);
  }
}

// 在某个时机（如页面加载、按钮点击时）调用这个函数
getUserInfo(123456);
```

### 装饰器中post请求和get请求的区别

#### POST和GET在URL上的唯一区别是：

 POST请求不会把主要的、要提交的数据放在URL的查询参数里。也就是登录时：

get请求时这样：  https:/129.80.32.33:8000/login?name="甘虎文"&password=1234。

post请求时这样：https:/129.80.32.33:8000/login。参数是隐形的，但它依然有一个清晰可见的URL。

```
1. 为什么要分POST和GET？
这主要是基于HTTP协议的设计哲学和语义。它们被设计用来做不同的事情：

GET：语义是 “获取” 数据。它应该是幂等的（多次执行效果相同）和安全的（不改变服务器状态）。因为它主要用于请求，所以参数放在URL里是合理的，方便缓存、收藏和分享链接。它的主要特点是“读操作”。

POST：语义是 “提交/创建” 数据。它通常不是幂等的，也不安全，因为它会改变服务器上的状态（比如创建新用户、下订单、发消息）。它的主要特点是“写操作”。

简单比喻：

GET 就像你在浏览器地址栏输入一个网址去查看一个网页。

POST 就像你填好一个表单，点击“提交”按钮去完成一个操作（如注册）。

将它们分开，符合“语义化”的设计原则，让网络请求的目的更清晰。


```

#### POST装饰器请求后的URL显示是怎样的？

假设我们有一个后端Flask路由：

```
@app.post('/api/user/login')  # 使用POST方法装饰器
def login():
    # 处理登录逻辑
    return {'success': True}
```

当你用前端代码（或Postman等工具）向这个接口发送一个POST请求时，你**仍然会看到这个URL**：
https://网站.com/api/user/login

**关键区别在于：**

- 如果是**GET**请求，你的用户名和密码可能会这样传递，**完全暴露在URL中**：
  `https://我的网站.com/api/user/login?username=甘虎文&password=123456` **(非常不安全！)**
- 而**POST**请求，URL依然是干净的 `https://你的网站.com/api/user/login`，但用户名和密码这些**敏感信息被放到了请求体（Body）里**，在标准的浏览器地址栏**是不可见的**。它们在网络传输中依然是明文（除非使用HTTPS加密），但不会出现在历史记录、服务器日志的URL部分等容易被看到的地方。

### 总结：

| 特性维度                | GET                                          | POST                                       |
| ----------------------- | -------------------------------------------- | ------------------------------------------ |
| **语义 (Semantics)**    | **获取**数据（Read）                         | **提交/创建**数据（Create）                |
| **幂等性 (Idempotent)** | **是**（多次执行结果相同）                   | **否**（多次执行可能产生不同结果或副作用） |
| **安全性 (Safe)**       | **是**（不应改变服务器状态）                 | **否**（会改变服务器状态）                 |
| **数据放置位置**        | URL 末尾的**查询字符串（Query String）**     | **请求体（Request Body）**                 |
| **数据可见性**          | 在**浏览器地址栏、历史记录、日志**中完全可见 | 数据在Body中，**不在地址栏显示**           |
| **数据长度限制**        | **有**（受限于URL最大长度，通常几KB）        | **理论上无限制**（服务器可配置限制）       |
| **缓存 (Cacheable)**    | **可被浏览器、代理服务器、CDN缓存**          | **默认不可缓存**                           |
| **书签/链接**           | **可收藏为书签，可被分享**                   | **不可收藏**（不包含数据体）               |
| **浏览器历史记录**      | 参数保留在历史记录中                         | 参数**不**保留在历史记录中                 |
| **浏览器刷新行为**      | 无任何警告，正常重新请求                     | 可能会弹出**“确认重新提交表单”**的警告     |
| **主要应用场景**        | 搜索查询、筛选数据、访问页面、点击链接       | 登录、注册、下单、支付、修改信息、上传文件 |
| **后端路由示例(Flask)** | `@app.get('/api/users')`                     | `@app.post('/api/users')`                  |
| **传输安全性**          | **HTTPS下，URL和参数均被加密**               | **HTTPS下，URL和Body均被加密**             |

------

### 核心结论与误区澄清

1. **首要区别是语义**：GET 用于“读”，POST 用于“写”。这是设计哲学和Web架构的基础，不应混淆。
2. **性能关键在缓存**：GET 可被缓存是提升Web性能的重要手段，POST 则不能。
3. post有body，可以携带字节流数据。get则不能，需要以？后跟参数的形式完成读操作。
4. **安全性的误区**：
   - **POST并不比GET更安全**：两者在HTTP下都是明文传输，在HTTPS下都同样安全加密。
   - **POST的“安全”优势**仅在于敏感参数不会直接暴露在地址栏、日志等可见的地方，降低了偶然泄露的风险，但**并未解决传输过程中的窃听问题**。解决窃听必须依靠**HTTPS**。



# 大模型训练和推理，哪一个成本更高

操作上：

训练：（170b模型为例）

1、前向传播：计算预测结果

2、反向传播：计算每个参数的梯度

3、参数更新：更新1700亿个参数

4、重复无数次，知道损失降到最小

推理：

前向传播：计算预测结果

硬件上：

都需要170b x 4字节=680GB的显存

推理只需要基础的170b的参数也就是680GB的显存 + 神经网络中每一个节点的激活

### 全量微调 vs. LoRA微调 vs. 模型推理 全面对比

| 维度           | 全量微调                                                     | LoRA微调                                                     | 模型推理                                                  |
| -------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | --------------------------------------------------------- |
| **核心概念**   | **重新教育**：更新模型所有参数以适应新数据。                 | **专项培训**：只训练少量新增参数，冻结原模型。               | **学以致用**：利用已训练好的模型进行预测。                |
| **目标**       | 让模型在特定领域或任务上获得最高性能。                       | 以较低成本使模型适应新任务，保持其通用能力。                 | 提供服务，产生实际价值。                                  |
| **计算复杂度** | **极高 (O(n³))**：涉及繁重的反向传播和所有参数的优化。       | **低 (O(n²))**：仅计算新增小参数量的梯度。                   | **中等 (O(n²))**：只需一次前向传播。                      |
| **内存消耗**   | **巨大**：需存储模型参数、梯度、优化器状态（如Adam的动量和方差）。 | **很小**：主要存储原模型，只需为新增的LoRA参数保留优化器状态。 | **固定**：只需加载模型权重和计算时的激活值。              |
| **硬件需求**   | **非常高**：需要多张顶级GPU（如A800/H100）进行分布式训练。   | **低**：通常单张或少量GPU即可完成。                          | **灵活**：从消费级显卡到GPU集群，取决于模型大小和并发量。 |
| **时间成本**   | **数天至数周**                                               | **数小时至数天**                                             | **毫秒至秒级**（ per request）                            |
| **资金成本**   | **极高（数十万至数百万美元）**                               | **低（数十至数千美元）**                                     | **按使用量计费**，单次请求成本极低。                      |
| **输出产物**   | **一套全新的、独立的模型权重**（体积与原模型相同）。         | **一个额外的、极小的适配器文件**（几MB到几百MB）。           | **预测结果**（如文本、分类标签、生成图片）。              |
| **主要优势**   | 性能潜力最高，可能达到领域最优。                             | **成本效率极高**，训练快，无灾难性遗忘，易于切换和部署。     | **即时产生价值**，可规模化服务，技术门槛低。              |
| **主要劣势**   | **成本巨高**，可能导致“灾难性遗忘”，存储和部署多个全量模型负担重。 | 性能上限可能略低于全量微调（但通常差距很小）。               | 无法定制模型行为，完全依赖于基础模型的能力。              |
| **技术门槛**   | **非常高**：需要深厚的分布式训练和深度学习工程经验。         | **中等**：有现成的库（如PEFT），流程相对标准化。             | **低**：有许多开箱即用的部署和服务框架。                  |
| **适用场景**   | 资金雄厚的巨头公司为构建核心壁垒而进行的基础模型训练或深度领域定制。 | **当前企业级应用的主流**：为不同客户、不同任务快速定制化大模型。 | 所有面向用户的AI应用和服务，如ChatGPT、Midjourney等。     |
| **比喻**       | **重造大脑**：为一项新技能重新连接所有神经元。               | **使用速记笔记本**：大脑不变，但准备了一个小本子记录特定领域的知识和技巧。 | **使用大脑思考**：直接运用已有的知识和技能来解决问题。    |







