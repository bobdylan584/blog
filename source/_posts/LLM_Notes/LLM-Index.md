---
title: LLM-Index
date: 2024-11-22 21:21:51
tag: LLM_Notes
---



# LLM-Index

## 昨日回顾

**1、output parsers**

字符串解析器

列表解析器

json解析器

pydantic解析器

自定义解析器

**2、memory**

ChatMessageHistory

​	history.add_user_message(xxx)

​	history.add_ai_message(xxx)

​	message_to_dict()

​	messages_from_dict()

ConversationChain(自动管理上下文)

​	ConversationChain（llm=model）

​	conversation.predict(input='xxx')

**3、Index(RAG核心组件)**

文件加载器

​	创建UnstructuredLoader对象load

​	docs = loader.load

​	html可以用自己的html对象

文档分割器

​	创建文档分割器的对象（separator,chunk_size,chunk_overlap）

​	单文档切割

​	多文档切割(打印信息不同，打印出多个document的key，看不到具体内容)

vectorstores

​	创建向量数据库Chroma（存入的文档，存入的路径，embedding）

​	加载

​	查询（要查询的问题，和匹配的文档k数）

检索器（和similarity search方法很相似	 	retriever=chromadaDB.as_retriever(search_kwargs={k:2})

​	result=retriever.invoke(query)

​	可以设置检索时候的算法

**agent**

基于大模型对用户的问题进行思考，并调用工具完成任务：

​	调用api（tools，model，AgentType.ZERO_SHOT_REACT_DESCRIPTION,verbose=True）

REACT(reason推理+act（调用工具完成任务）)循环往复。

tools=load_tools([wikipedia,'llm-math'],)

**每一种类型（大概的理论，怎么调用，怎么改动）**

项目为什么要调用哪些api

哪些组件、具体作用、使用方式（api）、知道如何修改

# RAG项目实战

智能衣答系统（RAG)

物流行业信息咨询智能问答系统（RAG)

目标：用langchain搭建一个系统

解决问题：模型基于过去的经验数据完成训练，

基于企业自有知识微调

基于基于langchain

**流程：**

`索引`（数据预处理：文档读取进来，提取内容成字符串，切割成chunk，embedding chunk）

`检索`：（用户提问题，转换问题为向量embedding，用embedding做相似度（知识向量库里的）匹配  （用向量库做一个能匹配k个答案的检索器）

`生成`：匹配到top_K个,添加到prompt，提交给LLM生成答案

加载文件、读取文件、文本分割、文本向量化、问题向量、匹配出的k个，匹配出文本作为上下文和问题一起添加到prompt。

**智能衣答系统（RAG)**

`1、项目需求`

以一副属性构建本地知识

`2、项目思路`

离线部分

​	文件加载

​	文本切分

​	向量化

​	存向量库

在线部分

​	query向量化

​	在文本向量中匹配出与问句向量相似的topk个

​	匹配出的文本作为上下文和问题一起添加到prompt中

​	提交给llm生成答案



```python3
P02_project
	data
    project1
    	db.py
    	
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from P04_RAG.P02_project.project1.model import embedding

def create_db(file_path,chunk_size=150,chunk_overlap=30,persist_directory='./chroma_data')
	第一步：加载文档
	loader = TextLoader('../data/衣服属性.txt',encoding='utf-8')
	docs = loader.load()
	print(docs)
	第二步：切分文档
	text_splitter = CharacterTextSplitter(separator="\n",chunk_size=150,chunk_overlap=30)
	split_texts  = text_splitter.split_document(docs)
	第三步：将切好的文档存储到向量数据库中(只执行一次即可，下次运行就从模型中加载即可)

	Chroma.from_documents(documents=split_texts,
                     	embedding=embedding,
                     	persist_directory=persist_directory)


def get_retriever(embedding,k=2,persist_directory='./chroma_data'):
	从构建好的向量数据库中加载数据
	vectordb = Chroma（persist_directory='./chroma_data',embedding_function=embedding)
	第四步：生成一个检索器
	retriever = vectordb.as_retriever(search_kwargs={'k':k})
	return retriever
	
	
create_db(file_path,chunk_size=150,chunk_overlap=30,persist_directory='./chroma_data'
retriever=get_retriever(embedding,k=2,persist_directory='./chroma_data')


```

```
model.py

from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from dotenv import load_dotenv
### 加载环境变量
load_dotenv()

### 加载embedding模型
embedding = DashScopeEmbeddings()

### 加载对话的大模型
llm = ChatTongyi(model='qwen-plus')
```

```
main.py



from langchain_core.prompts import PromptTemplate

from P04_RAG.P02_project.project1.db import get_retriever
from P04_RAG.P02_project.project1.model import embedding, llm

1、获取检索器
retriever = get_retriever(embedding)
def qa(question):
	2、将问题送入检索器，获取相似的top-k个文档
	
	docs = retriever.invoke(question)
	print(f'docs-->{docs})
	related_docs = [doc.page_content for doc in docs]
	print(f'related_docs-->{related_docs}')


	3、设置提示词模版，包括检索到的文档和用户的问题
	template_str = '''
	你是一个问答任务助手，使用以下检索到的上下文来回答问题，如果你不知道答案，只需说不知道，回答保持简洁。'''
	4、构造提示词
    prompt = template.format(question=question,context=related_docs)
	
	5、通过ChatMessageHistory()来存储上下文信息
	chat_history = ChatMessageHistory()
	chat_history.add_user_message(prompt)
	
	5、调用模型，生成答案
	result = llm.invoke(str(chat_history)).content
	print(f'result-->{result}')
	将大模型的输出保存到chat_history
	chat_history.add_ai_message(result)
	
	return result


if __name__ == __main__
question = '我身高170，体重140斤，买多大尺码'
retult = qa(question)

question = '我的体重是多少'
result = qa(question)

```

```
Conversational
condense question prompt:用于问题“重写”的提示模板
chain type链；stuff：全部塞进去给LLM(简单包里，’map reduce‘：先逐段问，再总结；)

把nexus584问大模型
prompt：nexus584的用户在7月末在csdn上发布了两篇文章，请把他的文章链接给我

chathistory
总结：使用ConversationalRetrieverchain
提示词模版+问题
原始的提示词；大模型的提示词（把对话信息加载进来了）


```

实现前端问答系统界面

```python3
import streamlit as st




st.title('智能衣答系统')
st.chat_input()
response = chat_history2[-1][1]

初始化会话状态，如果没有则创建

使用session state方式来储存对话历史，也就是将其设置为全局变量

user input



在页面上显示模型生成的回复

```

读取pdf，可以用不同的文档加载器的api ；called as PDF