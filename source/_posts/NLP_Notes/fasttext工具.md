---
title: fasttext工具
date: 2024-02-25 22:45:19
tag: NLP_Notes
---

# fasttext工具

## 1 fasttext工具

### 1.1 介绍

- 概念
    - 是一种文本分类和词向量训练的高效工具
- 作用
    - 文本分类 (分类模型)
    - 训练高质量词向量 (词嵌入模型)
- 特点
    - 高效, 快
    - 适用于大规模数据集

### 1.2 架构(了解)

- fasttext模型组成
    - 输入层 
        - 词向量 -> 根据词和词子词信息  词:apple 子词:app ppl ple
        - skipgram模型
        - CBOW模型
    - 隐藏层
        - 加权求和 -> 文本向量表示
    - 输出层
        - 文本分类
        - 线性层
        - softmax层
- 层次softmax
    - 由霍夫曼二叉树组成
    - 二叉树转换成是否问题 二分类问题
    - 树路径越短, 词概率越大; 树路径越长, 词概率越小
    - 层次softmax最多只需要计算 $$log_2词数$$ 次数, 普通的softmax计算 词数 的次数
- 负采样 
    - 将输出层的神经元分为正负两类, 正例神经元1个, 其余都是负例神经元
    - 在负例神经元中随机选择2-5个/5-20个进行反向传播
    - 其他Bert/GPT模型对所有的神经元进行反向传播

### 1.3 文本分类

- 概念: 将输入文本分成指定类型

    - 二分类
    - 多标签分类
    - 单标签分类

- 代码实现

    ```python
    import fasttext
    
    
    def dm01():
    	# 模型训练
    	model = fasttext.train_supervised(input='../data/cooking.train')
    	print(model)
    	# 模型预测
    	print(model.predict("Which baking dish is best to bake a banana bread ?"))
    	# 模型评估
    	print(model.test('../data/cooking.valid'))
    	
    # 数据预处理调优
    def dm02():
    	model = fasttext.train_supervised(input='../data/cooking.pre.train')
    	# 模型评估
    	print(model.test('../data/cooking.pre.valid'))
    	
    # 训练轮次调优
    def dm03():
    	model = fasttext.train_supervised(input='../data/cooking.pre.train', epoch=25)
    	# 模型评估
    	print(model.test('../data/cooking.pre.valid'))
    	
    # 学习率调优
    def dm04():
    	model = fasttext.train_supervised(input='../data/cooking.pre.train', epoch=25, lr=1.0)
    	# 模型评估
    	print(model.test('../data/cooking.pre.valid'))
    	
    # n-grams调优
    def dm05():
    	model = fasttext.train_supervised(input='../data/cooking.pre.train', epoch=25, lr=1.0, wordNgrams=2)
    	# 模型评估
    	print(model.test('../data/cooking.pre.valid'))
    	
    # 层次softmax调优
    def dm06():
    	model = fasttext.train_supervised(input='../data/cooking.pre.train',
    	                                  epoch=25,
    	                                  lr=1.0,
    	                                  wordNgrams=2,
    	                                  loss='hs')
    	# 模型评估
    	print(model.test('../data/cooking.pre.valid'))
    
    # 多标签多分类  loss='ova'  one vs all
    def dm07():
    	model = fasttext.train_supervised(input='../data/cooking.pre.train',
    	                                  epoch=25,
    	                                  lr=0.2,  # 学习率太大,达到最佳模型
    	                                  wordNgrams=2,
    	                                  loss='ova')
    	# 模型预测
    	# k: 返回的标签数量
    	# threshold: 返回大于等于预测概率阈值的标签
    	print(model.predict("Which baking dish is best to bake a banana bread ?", k=3, threshold=0.5))
    	
    # 自动搜索最佳参数模型
    def dm08():
    	model = fasttext.train_supervised(input='../data/cooking.pre.train',
    	                                  autotuneValidationFile='../data/cooking.pre.valid',
    	                                  autotuneDuration=600)
    	
    
    # 模型保存与加载
    def dm09():
    	model = fasttext.train_supervised(input='../data/cooking.pre.train',
    	                                  epoch=25,
    	                                  lr=1.0,
    	                                  wordNgrams=2,
    	                                  loss='hs')
    	# 保存
    	model.save_model(path='../data/cooking.bin')
    	# 加载
    	model = fasttext.load_model(path='../data/cooking.bin')
    	print(model.predict("Which baking dish is best to bake a banana bread ?"))
    if __name__ == '__main__':
    	# dm01()
    	# dm02()
    	# dm03()
    	# dm04()
    	# dm05()
    	# dm06()
    	# dm07()
    	# dm08()
    	dm09()
    ```

### 1.4 训练词向量

- 使用自/无监督学习训练语言模型
- 大量语料库训练出的词嵌入模型

### 1.5 词向量迁移

- 在特定领域使用已有的词嵌入模型

- 开箱即用

    ```python
    import fasttext
    
    
    # 词向量模型迁移: 在特定领域使用现成的向量模型
    # 加载词向量模型
    model = fasttext.load_model('../data/cc.zh.300.bin')
    print(model)
    
    # 句子先分词, 将词转换成词向量
    print(model.get_word_vector('音乐'))
    
    print(model.get_sentence_vector('hello world!'))
    ```

## 2 迁移学习

- 概念
    - 使用源任务(其他领域)模型去改进相关目标任务(下游领域)
    - 知识迁移: 使用相关领域的知识来优化下游领域任务
    - 什么情况使用迁移学习?
        - 源任务和目标任务相关性
        - 目标任务的数据集教少
- 预训练模型 pre-model
    - 在大量通用数据集上训练得到的模型
    - 预训练模型适用于大多数任务
    - 基座模型 -> 耗时, 需要的计算资源多
- 微调 fine-tuning
    - 基于下游任务的数据集对预训练模型进行优化
    - 全参微调
    - 部分参数微调
        - 冻结embedding层
        - 冻结encoder/decoder某些层

## 3 transformers库使用

### 3.1 transformers库是什么

- 收集预训练模型的开源库
- 各种开源大模型以及数据集
- 访问[https://huggingface.co](https://huggingface.co/)需要科学上网

### 3.2 transformers库使用

```properties
# 创建虚拟环境
conda create --name 虚拟环境名称 python=3.10
# 切换虚拟环境
conda activate 虚拟环境名称
# 安装transformers库
pip install transformers -i https://mirrors.aliyun.com/pypi/simple/
# 安装datasets库
pip install datasets -i https://mirrors.aliyun.com/pypi/simple/
# 安装torch cpu/gpu  当前是cpu版本
pip install torch -i https://mirrors.aliyun.com/pypi/simple/
```

- 管道方式

    - 文本分类任务

        ```python
        import torch
        from transformers import pipeline
        import numpy as np
        
        
        # 文本分类任务
        def dm01():
        	# 加载预训练模型
        	# 加载本地模型
        	# 加载在线模型 techthiyanes/xxx
        	# task: 任务名固定的, 不是自定义
        	model = pipeline(task='sentiment-analysis', model='../model/chinese_sentiment')
        	# 模型推理
        	result = model('我爱北京天安门，天安门上太阳升。')
        	print('result--->', result)
        
        
        if __name__ == '__main__':
        	dm01()
        ```

    - 特征提取任务

        ```python
        # 特征提取, 不带任务头, 等同于词嵌入模型
        def dm02():
        	model = pipeline(task='feature-extraction', model='../model/bert-base-chinese')
        	# 模型推理
        	output = model('人生该如何起头')
        	# [cls]xxxx[sep]xxx[sep]
        	# cls->整个序列语义表示
        	print('output--->', type(output), np.array(output).shape, output)
        
        
        if __name__ == '__main__':
        	dm02()
        ```

    - 完型填空任务

        ```python
        # 完型填空任务 MLM BERT预训练模型的子任务
        def dm03():
        	model = pipeline(task='fill-mask', model='../model/bert-base-chinese')
        	# 模型推理
        	# [MASK]: 掩码token表示
        	output = model('我想明天去[MASK]家吃饭。')
        	print('output--->', output)
        ```

    - 阅读理解任务

        ```python
        # 问答 阅读理解
        def dm04():
        	model = pipeline(task='question-answering', model='../model/chinese_pretrain_mrc_roberta_wwm_ext_large')
        	# 准备数据
        	context = '我叫张三，我是一个程序员，我的喜好是打篮球。'
        	questions = ['我是谁？', '我是做什么的？', '我的爱好是什么？']
        	# 模型推理
        	output = model(context=context, question=questions)
        	print('output--->', output)
        ```

    - 文本摘要任务

        ```python
        # 文本摘要
        def dm05():
        	model = pipeline(task='summarization', model='../model/distilbart-cnn-12-6')
        	context = "BERT is a transformers model pretrained on a large corpus of English data " \
        	          "in a self-supervised fashion. This means it was pretrained on the raw texts " \
        	          "only, with no humans labelling them in any way (which is why it can use lots " \
        	          "of publicly available data) with an automatic process to generate inputs and " \
        	          "labels from those texts. More precisely, it was pretrained with two objectives:Masked " \
        	          "language modeling (MLM): taking a sentence, the model randomly masks 15% of the " \
        	          "words in the input then run the entire masked sentence through the model and has " \
        	          "to predict the masked words. This is different from traditional recurrent neural " \
        	          "networks (RNNs) that usually see the words one after the other, or from autoregressive " \
        	          "models like GPT which internally mask the future tokens. It allows the model to learn " \
        	          "a bidirectional representation of the sentence.Next sentence prediction (NSP): the models" \
        	          " concatenates two masked sentences as inputs during pretraining. Sometimes they correspond to " \
        	          "sentences that were next to each other in the original text, sometimes not. The model then " \
        	          "has to predict if the two sentences were following each other or not."
        	# 模型推理
        	output = model(context)
        	print('output--->', output)
        ```

    - NER任务

        ```python
        def dm06():
        	model = pipeline(task='ner', model='../model/roberta-base-finetuned-cluener2020-chinese')
        	# 模型推理
        	output = model.predict('我爱北京天安门，天安门上太阳升。')
        	print('output--->', output)
        ```

- 自动模型方式

    - 文本分类任务
    - 特征提取任务
    - 完型填空任务
    - 阅读理解任务
    - 文本摘要任务
    - NER任务

- 具体模型方式

    - 完型填空任务

## 4 中文文本分类案例

- 任务介绍
- 加载数据集
- 创建数据加载器
- 自定义下游任务网络模型
- 模型训练
- 模型推理































