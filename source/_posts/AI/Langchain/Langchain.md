---
date: 2025-05-19 16:58:35
title: Langchain
categories: [AI, langchain]
tag: AI
---

# langchain介绍

## 概念

### 什么是LangChain
LangChain由 Harrison Chase 创建于2022年10月，它是围绕大模型应用的开发建立的一个框架。

LLMs使用机器学习算法和海量数据来分析和理解自然语言。GPT3.5、GPT4是LLMs最先进的代表，国内百度的文心一言、阿里的通义千问也属于LLMs。

LangChain自身并不开发LLMs，它的核心理念是为各种LLMs实现通用的接口，把LLMs相关的组件“链接”在一起，简化LLMs应用的开发难度，方便开发者快速地开发复杂的LLMs应用。

### 官网链接

https://python.langchain.com/docs/integrations/text_embedding/huggingfacehub

### langchain组件

LLM、Prompts、Memory、Index、Chains、Agent

#### 核心组件

LLM、Prompts、Memory

可以参考deepseek对话网页的主要组成部分：右上方的**大模型对象**、右下方的**聊天框**、左边的**会话记录**

一个大模型的应用部署，实际只需要一个大模型（调用api key、本地部署大模型）、一个提示词输入界面（对话框）、一个会话记录（让单次对话变成连续对话的“记忆体”。）

#### 增强能力的组件

##### Indexes (索引)

 这不是一个基础“操作”组件，而是一个数据准备和检索系统。它能将外部数据（各类文件）有效地提供给LLM。可以看作是为LLM准备的“外部知识库”或“工具箱”。

在**RAG系统**中，Index（索引）组件扮演着核心的“知识库”角色，是整个系统的基石。没有它，RAG就失去了检索外部知识的能力，退化为一个普通的、仅依赖模型内部知识的对话系统。

R(Retrieval)：Index 用于检索。

A(Augument)：Prompts 用于拼接query和检索到的上下文。

AA(Advanced Augument)：(Prompts+Memory)第二次的query+检索到的内容+上一次生成的答案，这里涉及到了上下文，需要用到memory组件。

G(Generation)：Models 用于生成答案。

Chains 通常被用来将“检索 -> 增强 -> 生成”这三个步骤固化为一个可重复执行的流水线。

 ##### Tools (工具)

任何特定的功能（如搜索、计算、查数据库）都被抽象成一个“工具”。Agent的核心就是使用工具。Chains也可以调用工具。

这些组件扩展了核心基石的能力，使其能处理更复杂的任务。

#### 组件的集成

Chains (链): 确定性的编排模式。将多个步骤（调用LLM、使用工具、查询索引）固定地串联起来。

Agents (智能体): 非确定性的编排模式。将LLM作为大脑，动态地决定调用哪些工具（或Chains）、查询哪些索引，并利用Memory来逐步完成任务。

### LangChain核心包

#### langchain-core

 - 基础抽象、LCEL、标准化接口
 - 适用于：自定义组件或底层扩展

#### langchain-community

 - 第三方集成（LLM、工具等）
 - 适用于：使用非官方或社区维护的集成

#### langchain

 - 高级链、代理、检索逻辑
 - 适用于：快速构建应用层逻辑（如 RAG、Agent）

# 作用

# 

## 

### 

#### 

