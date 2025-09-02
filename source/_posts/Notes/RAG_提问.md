---
date: 2025-04-05 00:27:31
title: RAG项目梳理
categories: [AI_Module, RAG项目梳理]
tag: AI_Module
---

# # 听课笔记

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



## 自问自答

### RAG是什么？

RAG（retrieve agument generation）检索增强生成。

检索：构建milvus、redis、mysql等数据库，以供查询。

增强：检索到和query相关的上下文，结合每一轮的query提示词，一直增加在history模块中，然后每次提问，都结合history数据，一起提供给大模型去生成。

生成：大模型作为大脑，结合问题和相关，思考生成回答。

### 为什么milvus数据库需要docker环境下安装，而mysql不需要。

不是MySQL“不需要”Docker，而是MySQL的传统部署方式已经做得太好，好到在生产环境中没有给Docker留下取代它的理由。

而像Milvus这样的新兴数据库，其传统部署方式异常复杂，反而让Docker成为了它的“标准”安装方式。

### milvus数据库中的索引，是什么意思？

这里的索引特指为了高效地进行向量相似性搜索而建立的一种数据结构。

目标是牺牲召回率，解决暴力搜索计算速度慢的问题，通过算法将向量空间进行划分、压缩和编码，建立索引。让检索系统，只检索最近的那个区域里的数据。相当于将图书馆的书，进行分门别类的整理。索引=类别=簇

方法：FLAT（暴力检索）、IVF系列（IVF_FLAT分簇、IVF_PQ量化、IVF_PQ）

相似度计算方法：L2欧式、IP点积，cos余弦相似度计算

### 多少条数据，才值得用ivf，而不是暴力搜索？

考虑的因素：

数据量：一万、十万、百万

向量维度：100、500、1000

性能要求：单个查询相应10ms以内，吞吐量：并发量。

准确率要求：

​	nprobe值越小，探查的簇就越少，搜索越快，召回率越低。可能错过真正的最近邻答案。

​	值越大，探查的簇越多，搜索越慢，召回率高，接近暴力搜索的结果。

### nprobe；这是词是哪几个单词组成的？

n代表 数量，probe代表 探查，合起来就是要检索的最相关的top-n 个探查点。例如，构建索引时，IVF会通过k means算法聚类成nlist个聚类中心。nlist就是列表的数量。

当query向量进来时，系统先计算它与所有nlist个聚类中心的距离，选出最近的nprobe个聚类中心，在这nprobe个聚类中心里面做暴力检索，计算query向量和该簇内所有向量之间的距离。

nprobe可以控制暴力检索的范围。nlist可以给数据分区。

nprobe=nlist；代表探查所有单元，此时IVF索引退化为暴力搜索。

nprobe（要探查的簇的数量）：是一个在搜索速度和结果精度（召回率）之间进行权衡的关键调优参数。

### IVF单词的组成是哪些？

IVF：inverted vector file （倒排向量文件）

比喻理解：

正排：通过书名找章节（关键字关键字是：深度学习，通过找所有的书，找到所有的章节）

倒排：通过章节找书名（关键字是：深度学习，找出所有包含“深度学习章节”的书）

### 聚类中心向量的计算公式

求所有向量的平均值；聚类中心向量=簇内所有向量相加，然后除以向量的个数。query先跟簇类中心向量做相似度计算。再和最相关簇类中心的所有向量进行相似度计算。

### k-means怎么聚类的？

先自设置n个中心点，然后每个向量距离这个中心点做一个欧式距离的距离计算，距离最近的就属于这个类，然后按照这个类里面的向量，重新计算簇心。

也就是先在数据样本中，随机定三个簇心，完成分簇。簇内再计算平均值作为新簇心。这个新簇心可能不是任何一个样本。

### 请帮我解释一下，docker容器中的milvus-redis、milvus-standalone、milvus-minio、milvus-redis、milvus-etcd，分别是什么意思，请解释一下？

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