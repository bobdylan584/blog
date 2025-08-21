---
title: RAG流程
date: 2025-01-07 21:21:51
tag: RAG_Notes
---



# RAG流程：

mysql储存FQA高频问答对数据

问题检索：BM25

连接数据库

添加表（带着字段）

添加数据

json.dumps()

mysql 存储自己的网址和密码。（自己设计一个RAG系统）

声明回退问题，把原来的复杂查询简化，第一个query检索

进行改写。主题含义不变。

milvus可以处理的数据集的大小限制是多少

技术实现：

增强索引：设计目标、核心功能、技术实现

多粒度切块，把块-分子块，对应的父块，提供给LLM

文档切成一块，存储milvus中的文档，

query是为题，编程向量，

太长的拆成四个

128个向量

父块是一个

子块分成子块去做检索

切块的子块数都是超参数

混合检索：BM25，向量检索，字符检索

base：基础模块，配置、日志

core：核心逻辑模块，实现RAG的关键功能

main：系统运行入口，支持数据处理和交互查询

中午将一份唯二

通用知识由大语言模型回答，

直接 hyde 子查询 会输

文档检索：支持抽向量和系数向量的混合检索，

中午，下午

语义关键字，倒排（关键字检索

两句话的相似性，

混合检索，重排序优化，

作为回答送给大模型，方便理解。

用户查询

代码目录结构：

配置管理、日志记录

config。py

最大支持customer service phone

fallback

document_process

langchain的文档加载器

markdowm text splitter

datetime import datetime

相对路径（三方包）

模型切分工具、

文档加载器的类（处理pdf、word、ppt、图片

OCR可以提取图像里的内容

optical character recognition光学字符识别

paddle paddle ocr的工具库，基于深度学习技术，

可以把图像中的文字提取出来。

pdf 中的图片，怎么解决？paddle OCR；只能识别简单的图片rapid OCR

cv2：

寻味羊*村超BA之苗侗味道（第一档口华兴美食城店）

docx第三方库

迭代器

读取ppt的内容OCRIMGLoader（file path）

添加源数据，direcotry_path

documents 所有的键

扩展名集合

source