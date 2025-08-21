---
title: RAG-Langchain
date: 2025-01-04 01:21:51
tag: RAG_Notes
---



# RAG-Langchain

## RAG

解决什么问题：

信息过时：网络检索，获取最新数据

领域知识缺失：微调，将专有和私有的知识放到知识库里

幻觉：RAG（retrieval augmented generate），减轻幻觉，基于相关文档进行生成，

安全：RAG,无需将数据送到公开大模型中训练，放到本地知识库，使用本地的模型（api会泄露）进行调用，避免数据的公开和泄露。2）私有数据时存在本地知识库的，做一个权限的管控。

RAG定义：检索技术+生成（LLM提示）

处理流程：构建索引（文件加载、内容读取、chunk构建（拆成小文件，小块）、向量化（小块文档向量化）、落向量化

检索：query向量化，找到topk

生成：topk+query构建prompt；llm生成。

开发框架：LLaMAIndex、Langchain（快速搭建大模型）

## Langchain

langchain将模型分为三种（

langchian是用于构建大模型应用程序的框架，帮助开发者更高效的组合和使用多语言的工具。

原始大模型：LLM、chat models、embeddings

chain：组装chain：chain=LLMChain（llm=model,prompt=prompt_template)

## output_parsers

字符串解析器、列表解析器、json解析器、定义类、自定义解析器

根据实际情况，调用api，做规范化

字符串解析器：提取模型返回的原始文本

```python3
创建简单链
创建字符串解析器
stroutputparser()
组合组件
prompt|model
调用链
result=chain.invoke()
将字符串解析器添加到链中

```

列表解析器：文本转换为列表

parser.get format instructions 创建带格式说明的提示模版

```
创建列表解析器
创建提示模版，包含列表解析器的格式说明，也就是你要告诉模型输出的格式，然后大模型会按照这个输出结果

```

Json解析器

```
创建带格式说明的提示模版
json.parse.geet_format_instructions()
Chatprompttemplate.from_template()
```

Pydantic解析器：python库，用于数据验证

```
定义pydantic模型
创建pydantic解析器
创建带格式说明的提示模版
组合组件
调用链
```

自定义解析器（输出格式非常复杂情况下，就自定义格式进行输出）

```
需要继承baseoutputparser，实现parse（）方法，get format instructions（）方法
解析大模型的输出，然后组织成想要的个数
提供格式指导给模型
处理空白和换行符，用冒号做split切分；返回一个字典
custom，
定义大模型输出的格式
from template（模版说明）

```

解析器：告诉大模型该怎么输出 get_fromat_instructions()

invoke(填充好提示词)，最后通过结果解析器，调用parse方法

核心是实现parse方法。

## Memory

储存上下文，储存历史

langchain提供了memory组件：

ChatMessageHistory:只是简单存储

```python3
history = ChatMessageHistory（）
添加用户消息
history.add_user_message('xxx')
添加大模型回复的消息
history.add_ai_message('xxx')

```

ConversationChain：

创建一个对象，放一个llm=llm参数

```
实例化一个对象，传一个大模型
model = ChatTongyi（model='',temperature=3000)
实例化会话链，这里需要传入模型
conversation=ConversationChain（llm=model）
conversation.predict(input='123')
conversation.predict(input='456')
conversation.predict(input='一共有几个数字')
print(result)
conversation自动维护上下文信息
```

## indexes

让langchain具备处理文档处理的能力，包括文档加载、检索、文档加载器、文本分割器、vectorstores、检索器。

**文档加载器**

```python3
引入loader工具
初始化对象
loader = UnstructureLoader('./data/衣服属性.txt,encoding='utf8')
加载数据到内存中（返回值是一个列表，每一个元素是一个document对象，对应原文件的一行数据
docs = loader.load
print(f'len-->{len(docs)}')
打印第一行数据
docs[0].page_content
langchian_community：证明是三方写的包（相比之下，三方打印更简洁了）
from langchian_community.document.loader import Textloader
textloader返回值是一个列表，每一个元素是一个document对象，对应原文件的所有数据

```

**文档分割器**：分割字符，按照字符长度

chunk_size= 多少数量切一块；chunk_over=重叠字符数；separator=按照分割符进行切割。既可以制定分隔符。

e.g.假设chunk size=500；chunk over=0；separator=句号；文本中有五个句子。第一个句子100个字符，第二个句子200个字符，第三个句子100；不超过500，则以上三个句子打包成一个chunk。但是第四个句子是200个字符。则放弃第四个句子。第四放到第二个chunk中

```
langchain.text.splitters import CharacterTextSplitter
创建分词器separator参数指的是分割的分隔符，chunk size指的是分割出来的每个块的大小，chunk overlap是指每个块之间重复的大小
CharacterTextSplitter(separator="",chunk_size=5,chunk_overlap=0)
一句话进行分割
text_splitter.split_text('a b c d e f')
多句话分割（传一个可迭代对象，列表）
text_splitter.create_documents('a b c d e f','e f g h')
```

缺点：

按数量进行切割，无法保存完整语义

如果读取的文档中某一段分隔符缺失，或者长段句子没有分隔符，则依旧存在句子过长的问题

只适合简单文本，不适应复杂文本

## 递归字符文本分割器

可以使用多个分隔符，依次使用分隔符，知道长度满足要求为止

separators=['\n\n','\n',' ']

## 语义文档分割器

基于语义相似性分割文本

计算成本较高，

适用于需要高度语义理解的场景（文档比较重要的情况，网页爬取就不需要了）

embedding文本之后，判断语义相似性。

字符-递归-语义（随复杂和重要程度递增）

## 其他专用分割器（MarkdownHeaderTextSplit）

按照你的章节去做切割，一章内大概都是一个意思

先实例化对象

markdown_spliter=MarkdownHeaderTextSplitter()

markdown_text=

docs = markdown_splitter.split_text(markdown_text)

再用对象去做切割

自定义分割器

**chains回顾**

连接LLM与其他组件，完成应用程序的开发过程



## MESS

导演名称，年份、信息对称，返回的结果，使用dantic，从结果中提取出来。就是这个结果。看齐的类型。

**字符分割器（characterTextSplitter）**

字符分割器是基于，按照空格进行切分。按照指定的数量，递归字符文本分割器（recursive character textsplitter）

递归字符文本分割器是一种更智能的分割方法，它尝试在特定分割符处分割文本，以保持更好的语义完整性。

特点：

尝试在自然断点处分割文本

比简单的字符分割更能保持语义完整性

适用于结构化成都较高的文本，如markdown、html

运行流程：

首先尝试使用第一个分隔符（如“\n\n")分割文本

如果分割后的快任然过大，则使用给下一个分割符继续分割

重复此过程，知道达到指定的chunk_size或则用完所有的分隔符

**语义文档分割器**：更高级的分割方法

基于语义相似性分割文本

能够更好地保持语义完整性

计算成本较高，处理大量文本时可能效率较低

使用于需要高度语义理解的场景

其他专用分割器

如markdownHeadTextSplitter

**文本分割参数调优策略**

chunk size

chunk overlap

separator参数

**非文本**

sql lite（在本地存一个文件）

similarity search查询相应的文档

**检索器**

vectordb.similarity_search()用途：直接调用向量数据库的相似度搜索功能。输入：查询字符串（query）和可选的返回数量（k）。输出：按相似度排序的文档列表。

其他应用领域包括：

1、语义搜索系统

2、推荐系统

3、文档聚类

4、异常检测

5、多模态系统

6、内容去重

7、知识图谱

8、情感分析：捕捉文本的（情感特征）

检索器的功能和vectordb.similarity_search()相似；

调用检索器的方法：具体实现方式不一样。

可以封装到langchain的chain里面。

检索器是langchain中负责信息检索的模块，通常与索引（indexes）模块（如向量存储、文档加载器）结合使用。它的核心功能是：

输入：接受用户查询（文本）

处理

输出：返回一组相关文档或文本片段

工作流程：

查询嵌入

相似性搜索

文档返回

后处理：对查询结果做一些加工，（返回k个文档，对k个文档进行排序、过滤或重新排名）

检索器的核心依赖：

嵌入模型：

向量储存：

相似性度量：

similarity：精确匹配；最相关的k个做一个返回

mmr：平衡相关性和多样性；返回k个；生成综合性报告

similarity score threshold 质量过滤； 动态数量； 不控制多样性；高精度筛选；

lambda_mult:控制多样性（仅MMR搜索有效）

vectordb as_retriever()

返回一个retriever对象，用于langchain链式调用

tf-idf retriever

MULTIQUERYRETRIEVER

Contexttual 

使用语言模型查询，生成多个解锁器。

ensembel retriever

查询依旧，但是检索不唯一。两种，或者以上。向量检索，达到非常好的效果。

向量检索的好。

custom retriever 生成多个查询语句，不同方式进行检索。把三种方法结合起来，进行使用。在里面实现。集成base retriever。看看试验方式，retriever；企业里常见的是混合优化，结构化优化。

## Agent

Agent是一种能够感知环境、进行决策和执行动作的智能实体。不同于传统的人工智能，Agent具备通过主动思考、调用工具去逐步完成给定目标的能力。

外部工具，保存语义理解和推理能力，任务规划能力

a=大模型+任务规划+执行（调用外部工具）+记忆

大脑、五官、四肢

点菜（任务是找吃的）；agent不知道，会调用美团、大众点评的api去搜附近美食，

自动给你点菜；最后在“付款”步骤，介入人工，避免支付风险。

用户提出任务：“agent启动；将输入和提示词模版结合，送给大模型

2、思考决策：大模型接受输入后，根据内置逻辑和提示词指导，进行思考

判断需要更多东西

是否需要工具

工具执行

结果反馈

循环

如果回答：当大模型判断任务已完成，无需额外工具即可回答时，它会生成最终的答案。

反思、自我批评、思考链、子任务分解

planning是计划，调用方法（搜索、调度、计算器的方法），完成规划里的任务

Action（一些需要调用，一些不需要调用外部工具）

先想（planing）再做；（循环）；再思考过程中，会产生上下文（放到memory里面）

任务规划（planning）；思考用户交代的任务

长短期记忆（memory）：聊天上下文，长期保留和回忆信息

langchain实现智能体

langchain提供了不同类型的代理（主要罗列了三种：）

zero-shot-react-description：代理使用react框架，仅基于工具的描述来确定

structured-chat-zero-shot-react-description结构化的参数输入

conversational-react-description：上下文保存到记忆里面。

agent_toolkits.load_tools import load_tools

from langchain.agent import initialize_agent

tools,llm,agenttype.zero_shot_react_description,verbose=True

提示词传入

agent.invoke()

initialize_agent(tools,model,agent_type.ZERO_SHOT_REACT_DESCRIPTION,VERBOSE=TRUE)



```python3
1、实例化大模型：
chatTongyi（model=‘qwen-max）
2、设置工具：
load_tools(wikipedia，llm-math，llm=model)
3、创建代理
initialize_agent(tools,model,agent_type.ZERO_SHOT_REACT_DESCRIPTION,VERBOSE=TRUE)
4、设置提示词并进行agent调用
prompt = 
result = agent.invoke(prompt)
print(f'result-->{result})
思考过程：思考-动作（循环）
finished chain
result-->

```

**总结：**

什么是langchain？本质是一个框架（目的是快速的智能应用的开发）

组件

models（嵌入模型（embedding model）、普通大模型、聊天大模型）

prompt（普通prompt、chatprompt）

chains（chain、LCEL)

output parsers（字符串解析器、列表解析器、json解析器、pydantic解析器）

memory（保存上下文信息，conversationchain..

indexes（文档加载器、文档分割器、向量存储）

检索器（基于问题，在知识库中检索出对应的答案）

Agent（智能体是什么？）