---
title: transformer介绍
date: 2024-02-08 09:03:41
tag: NLP_Notes
---

# day12_课堂笔记

## 1 transformer介绍

- 概念
  - transformer是基于自注意力机制的seq2seq模型/架构/框架
- 核心思想
  - 基于注意力机制
  - 自注意力
  - 一般注意力
- 作用
  - 捕获超长距离语义关系
  - 并行计算
  - 灵活性: 处理不同的数据, 文本/语音/图像/视频
  - 扩展性: 层数和多头数量可调, transformer默认是6层, 8个头

## 2 transformer架构

![1749118800244](1749118800244.png)

- 输入部分
  - 词嵌入层
  - 位置编码层
- 输出部分
  - 线性层
  - softmax层
- 编码器部分
  - 多头自注意力子层
  - 前馈全连接子层
  - 残差连接层
  - 规范化层(层归一化)
- 解码器部分
  - 掩码多头自注意力子层
  - 编码器-解码器堵头一般注意力子层
  - 前馈全连接子层
  - 残差连接层
  - 规范化层(层归一化)

## 3 输入

### 3.1 文本嵌入层

- 概念

    - 将token转换成词向量过程
    - nn.Embedding()

- 代码实现

    ```python
    # 输入部分是由 词嵌入层和位置编码层组成   x = word_embedding + position_encoding
    import torch
    import torch.nn as nn
    import math
    
    
    # 词嵌入层
    class Embeddings(nn.Module):
    	# todo:1- 定义构造方法 init
    	def __init__(self, vocab_size, d_model):
    		super().__init__()
    		# 初始化属性
    		self.vocab = vocab_size  # 词表大小
    		self.d_model = d_model  # 词向量维度
    		# 初始化词嵌入层对象
    		# padding_idx: 将值为0的值, 不进行词向量, 用0填充
    		self.embedding = nn.Embedding(num_embeddings=self.vocab,
    		                              embedding_dim=self.d_model,
    		                              padding_idx=0)
    	
    	# todo:2- 定义forward方法 前向传播
    	def forward(self, x):
    		# 词嵌入结果乘以根号维度数
    		# 最终的词向量值和后续位置编码信息差不太多, 实现信息平衡
    		# 后续注意力机制使用的缩放点积, 乘和除相抵消
    		return self.embedding(x) * math.sqrt(self.d_model)
    
    
    if __name__ == '__main__':
    	vocab_size = 1000
    	d_model = 512
    	# 创建测试数据
    	x = torch.LongTensor([[100, 2, 421, 508], [491, 998, 1, 0]])
    	# 创建词嵌入对象
    	my_embedding = Embeddings(vocab_size, d_model)
    	# 调用对象实现词嵌入
    	embedded_result = my_embedding(x)
    	print('embedded_result--->', embedded_result.shape, embedded_result)
    ```

### 3.2 位置编码器

- 概念

    - 通过一些计算方式给词向量引入位置信息工具
    - 位置编码器替代rnn/lstm/gru中的顺序执行 -> 拿到token和token位置信息

- 作用

    - transformer中不使用rnn/lstm/gru计算语义, 没有位置概念
    - 引入位置信息
    - x 我 x -> 没有位置信息:我x(前x) = 我x(后x) 引入位置信息:我x(前x) != 我x(后x)

- 方法

    - 使用正弦和余弦函数

        ![1749263379065](1749263379065.png)

    - 计算每个token在512维度上所有位置信息
    - PE -> 词维度上的位置信息
        - 索引下标偶数位用sin(第1,3,5...词), 索引下标奇数位用cos(第2,4,6...词) 
        - 我 -> [sin(), cos(), sin(), ...]
        - 爱 -> [sin(), cos(), ...]

- transformer中为什么要引入位置编码?

    - 注意机制计算时只算token和token之间的语义关系/相关性, 没有考虑位置关系, 需要引入位置编码
    - transformer中使用正弦和余弦函数计算位置信息值
        - 周期性 -> sin(α+β) = sin(α)cos(β) + sin(β)cos(α)
            - 我-> sin(10) -> sin(10+20) = sin(10)cos(20) + sin(20)cos(10) = 爱
            - 可以学习到我和爱之间的位置关系(规律)
        - 取值范围[-1,1], 避免值太大或太小, 导致梯度消失或爆炸

- 代码实现

    ```python
    # 位置编码器
    class PositionalEncoding(nn.Module):
    	# todo:1- init方法, 计算出n个词位置矩阵
    	def __init__(self, d_model, max_len=5000, dropout_p=0.1):
    		super().__init__()
    		# 初始化属性
    		self.d_model = d_model  # 词向量维度, 模型维度
    		self.max_len = max_len  # 句子最大长度
    		self.dropout = nn.Dropout(p=dropout_p)
    		
    		# 获取句子所有token索引下标,作为pos
    		# .unsqueeze(1)->在1轴升维 [[0],[1],...]
    		pos = torch.arange(0, self.max_len).unsqueeze(1)
    		# print('pos--->', pos.shape, pos)
    		# 创建一个pe全0矩阵, 存储位置信息  形状(句子长度, 词维度)
    		pe = torch.zeros(size=(self.max_len, self.d_model))
    		# print('pe--->', pe.shape, pe)
    		# 获取2i结果, 对向量维度d_model取偶数下标值
    		_2i = torch.arange(0, self.d_model, 2).float()
    		# print('_2i--->', _2i)
    		# 计算位置信息 奇数位的词sin  偶数位的词cos
    		pe[:, ::2] = torch.sin(pos / 10000 ** (_2i / self.d_model))
    		pe[:, 1::2] = torch.cos(pos / 10000 ** (_2i / self.d_model))
    		# 将pe位置矩阵升维, 三维数据集
    		pe = pe.unsqueeze(0)
    		# print('pe--->', pe.shape, pe)
    		# 存储到内存中, 后续便于加载
    		# pe属性中存储的是pe矩阵结果
    		self.register_buffer('pe', pe)
    	# todo:2- forward方法 将位置信息添加到词嵌入结果中
    	def forward(self, x):
    		"""
    		:param x: 词嵌入层的输出结果
    		:return: 编码器的输入x
    		"""
    		print('x--->', x.shape, x)
    		# x.shape[1], 句子中有多少个真实的token, 就在pe矩阵中取前多少个就可以
    		print('x.shape[1]--->', x.shape[1])
    		print('self.pe[:, :x.shape[1], :]--->', self.pe[:, :x.shape[1], :].shape, self.pe[:, :x.shape[1], :])
    		return x + self.pe[:, :x.shape[1], :]
    
    
    if __name__ == '__main__':
    	vocab_size = 1000
    	d_model = 512
    	# 创建测试数据
    	# max_len=60, 句子最大长度
    	# 当前第1个句子长度为4, 后续需要补56个0
    	# 当前第2个句子长度为4
    	x = torch.LongTensor([[100, 2, 421, 508], [491, 998, 1, 211]])
    	# 创建词嵌入对象
    	my_embedding = Embeddings(vocab_size, d_model)
    	# 调用对象实现词嵌入
    	embedded_result = my_embedding(x)
    	print('embedded_result--->', embedded_result.shape, embedded_result)
    	
    	# 创建pe位置矩阵 生成位置特征数据[1,60,512]
    	my_pe = PositionalEncoding(d_model=d_model, dropout_p=0.1, max_len=60)
    	# 调用位置编码对象
    	pe_result = my_pe(embedded_result)
    	print('pe_result--->', pe_result.shape, pe_result)
    ```

## 4 编码器

### 4.1 掩码张量

- 概念

    - 掩盖一些信息的二进制张量或下三角矩阵张量

- 作用

    - 屏蔽填充的信息
    - 屏蔽未来的信息

- transformer中使用

    - 编码器中只能使用padding mask
    - 解码器中使用padding mask 和 casual mask

- 代码实现

    ```python
    import torch
    import matplotlib.pyplot as plt
    
    
    # tril():生成下三角矩阵
    # triu():生成上三角矩阵
    # diagonal: 移动对角线
    def subsequent_mask(size):
    	# 下三角
    	causal_mask = torch.tril(torch.ones(size=(size, size)), diagonal=0)
    	# 上三角
    	# causal_mask = torch.triu(torch.ones(size=(size, size)), diagonal=0)
    	return causal_mask
    
    
    if __name__ == '__main__':
    	causal_mask = subsequent_mask(20)
    	print('causal_mask--->', causal_mask.shape, causal_mask)
    	
    	# 绘图
    	plt.figure()
    	plt.imshow(causal_mask)
    	plt.show()
    	
    	# 模拟自回归,进行自回归掩码
    	scores = torch.randn(size=(5, 5))
    	mask = subsequent_mask(5)
    	print('mask==0--->', mask==0)
    	masked_result = scores.masked_fill(mask==0, value=float('-inf'))
    	print('masked_result--->', masked_result)
    ```

### 4.2 自注意力机制

- 概念

    - 在同一序列中进行注意力计算
    - q=k=v

- 作用

    - 并行计算
    - 捕获更长距离的语义关系

- 为什么除以​$\sqrt{d_k}$

    - 防止`q*k^T`乘积值过大, 产生梯度饱和, 导致梯度消失

- 代码实现

    ```python
    from input import *
    
    
    # 定义缩放点积注意力规则函数, 方便后续调用
    def attention(query, key, value, mask=None, dropout=None):
    	"""
    	注意力计算封装函数
    	:param query: 输入x  解码器掩码多头自注意力子层输出
    	:param key: 输入x  编码器输出结果
    	:param value: 输入x  编码器输出结果
    	:param mask: 是否掩码
    	:param dropout: dropout层对象 函数名
    	:return: 动态c, 权重概率矩阵
    	"""
    	# todo:1- 获取d_k, 词维度数
    	d_k = query.shape[-1]
    	# print('d_k--->', d_k)
    	# todo:2- q和k计算权重分数矩阵
    	scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    	# print('scores--->', scores.shape, scores)
    	# todo:3- 判断是否需要进行掩码操作
    	if mask is not None:
    		scores = scores.masked_fill(mask == 0, value=-1e9)
    		print('='*50)
    		# print('scores--->', scores.shape, scores)
    	# todo:4- 权重分数矩阵进行softmax操作, 得到权重概率矩阵
    	p_attn = torch.softmax(scores, dim=-1)
    	print('p_attn--->', p_attn.shape, p_attn)
    	# todo:5- 判断是否对权重概率矩阵进行dropout正则化
    	if dropout is not None:
    		p_attn = dropout(p_attn)
    	# todo:6- 计算动态c矩阵
    	c = torch.matmul(p_attn, value)
    	print('c--->', c.shape, c)
    	return c, p_attn
    	
    if __name__ == '__main__':
    	vocab = 1000  # 词表大小是1000
    	d_model = 512  # 词嵌入维度是512维
    	
    	# 输入x 形状是2 x 4
    	x = torch.LongTensor([[100, 2, 421, 0], [491, 998, 0, 0]])
    	
    	# 输入部分的Embeddings类
    	my_embeddings = Embeddings(vocab, d_model)
    	embedded_result = my_embeddings(x)
    	
    	dropout_p = 0.1  # 置0概率为0.1
    	max_len = 60  # 句子最大长度
    	
    	# 输入部分的PositionalEncoding类
    	my_pe = PositionalEncoding(d_model, max_len, dropout_p)
    	pe_result = my_pe(embedded_result)
    	
    	# 调用attention函数
    	# 准备q,k,v
    	query=key=value=pe_result  # 自注意力
    	print('query--->', query.shape)
    	# 准备mask掩码张量 padding_mask
    	# unsqueeze(1) -> 形状(2,1,4) 后续进行masked_fill操作, 会进行广播, 变成 (2,4,4)
    	mask = (x!= 0).type(torch.uint8).unsqueeze(1)
    	# casual mask
    	# mask = torch.tril(torch.ones(4, 4))
    	print('mask--->', mask.shape, mask)
    	c, p_attn = attention(query, key, value, mask)
    	print('c--->', c.shape, c)
    	print('p_attn--->', p_attn.shape, p_attn)
    ```

### 4.3 多头注意力机制

- 概念

    - 使用多个头并行计算注意力, 可以从不同子空间维度学习特征

- 作用

    - 得到更丰富的特征
    - 增强模型表达能力

- 多头指的是什么

    - 在进行注意力计算时由多个人分别取算不同子空间的注意力

- 实现流程

    - q,k,v分别经过线性层计算, 得到wq,wk,wv

    - 将线性计算的q,k,v进行多头转换操作,分别计算注意力

    - 将多头注意力结果合并(类似于还原操作)后的结果经过线性层计算

        ![1749343420025](1749343420025.png)

- 代码实现

    ```python
    # 定义克隆函数, 用于克隆不同子层
    def clones(module, N):
    	return nn.ModuleList(copy.deepcopy(module) for _ in range(N))
    
    
    # 创建多头注意力机制类
    class MultiHeadedAttention(nn.Module):
    	# todo:1- init方法
    	def __init__(self, head, d_model, dropout_p=0.1):
    		super().__init__()
    		assert d_model % head ==0, 'd_model不能被head整数'
    		self.d_k = d_model // head
    		self.dropout = nn.Dropout(p=dropout_p)
    		self.head = head
    		# 初始为None, 还没有计算注意力
    		self.attn = None
    		# 4个线性层
    		# 前3个分别对q,k,v进行线性学习
    		# 第1个对多头注意力拼接结果进行线性学习
    		self.linears = clones(nn.Linear(d_model, d_model), 4)
    		# print('self.linears--->', self.linears)
    	# todo:2- forward方法
    	def forward(self, query, key, value, mask=None):
    		# todo:1- 获取batch_size大小
    		batch_size = query.size()[0]
    		# print('batch_size--->', batch_size)
    		# todo:2- 准备空列表, 存储线性计算+变形结果
    		output_list = []
    		# todo:3- q,k,v分别进行线性计算
    		for model, x in zip(self.linears, (query, key, value)):
    			# print('model--->', model)
    			# print('x--->', x)
    			output = model(x)
    			# todo:4- 线性计算结果变形 -> (batch_size, seq_len, head, d_k)
    			# transpose(1, 2):词数和词向量相邻, 更好的学习特征
    			output = output.view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
    			# todo:5- 将变形结果保存到空列表中
    			output_list.append(output)
    		# 获取q, k, v
    		# print('output_list--->', len(output_list))
    		query = output_list[0]
    		key = output_list[1]
    		value = output_list[2]
    		# todo:6- 计算多头注意力, 调用attention函数  (batch_size, seq_len, head, d_k)
    		x, p_attn = attention(query, key, value, mask)
    		# print('x--->', x.shape)
    		# todo:7- 多头注意力结果变形 -> (batch_size, seq_len, word_dim)
    		x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head*self.d_k)
    		# print('x--->', x.shape)
    		# todo:8- 经过线性层计算返回输出结果
    		# self.linears[-1]: 线性层对象
    		x = self.linears[-1](x)
    		return x
    	
    	
    if __name__ == '__main__':
    	vocab = 1000  # 词表大小是1000
    	d_model = 512  # 词嵌入维度是512维
    	
    	# 输入x 形状是2 x 4
    	x = torch.LongTensor([[100, 2, 421, 0], [491, 998, 0, 0]])
    
    	# 输入部分的Embeddings类
    	my_embeddings = Embeddings(vocab, d_model)
    	embedded_result = my_embeddings(x)
    
    	dropout_p = 0.1  # 置0概率为0.1
    	max_len = 60  # 句子最大长度
    
    	# 输入部分的PositionalEncoding类
    	my_pe = PositionalEncoding(d_model, max_len, dropout_p)
    	pe_result = my_pe(embedded_result)
    
    	# 调用attention函数
    	# 准备q,k,v
    	query=key=value=pe_result  # 自注意力
    	# print('query--->', query.shape)
    	# 准备mask掩码张量 padding_mask
    	# unsqueeze(1) -> 形状(2,1,4) 后续进行masked_fill操作, 会进行广播, 变成 (2,4,4)
    	# 多头注意力机制,需要得到(2,1,1,4)形状mask
    	mask = (x!= 0).type(torch.uint8).unsqueeze(1).unsqueeze(2)
    	# casual mask
    	# mask = torch.tril(torch.ones(4, 4))
    	# print('mask--->', mask.shape, mask)
    	# c, p_attn = attention(query, key, value, mask)
    	# print('c--->', c.shape, c)
    	# print('p_attn--->', p_attn.shape, p_attn)
    	
    	head=8
    	# 创建多头注意力机制类对象
    	my_mha = MultiHeadedAttention(head, d_model)
    	# 调用多头注意力机制对象
    	mha_result = my_mha(query, key, value, mask)
    	print('mha_result--->', mha_result.shape, mha_result)
    ```

### 4.4 前馈全连接层

- 概念

    - 由两层线性层和一层relu激活层
- 作用

    - 提取更丰富的非线性特征, 增强模型表达能力

- 代码实现

    ```python
    # 前馈全连接层
    class PositionwiseFeedForward(nn.Module):
    	def __init__(self, d_model, d_ff, dropout_p=0.1):
    		super().__init__()
    		# 定义两层线性层
    		# d_ff>d_model
    		self.linear1 = nn.Linear(d_model, d_ff)
    		self.linear2 = nn.Linear(d_ff, d_model)
    		# 定义dropout层
    		self.dropout = nn.Dropout(p=dropout_p)
    	
    	def forward(self, x):
    		output = torch.relu(self.linear1(x))
    		output = self.dropout(output)
    		print('output--->', output.shape)
    		return self.linear2(output)
    
    
    if __name__ == '__main__':
    	vocab = 1000  # 词表大小是1000
    	d_model = 512  # 词嵌入维度是512维
    	
    	# 输入x 形状是2 x 4
    	x = torch.LongTensor([[100, 2, 421, 0], [491, 998, 0, 0]])
    	
    	# 输入部分的Embeddings类
    	my_embeddings = Embeddings(vocab, d_model)
    	embedded_result = my_embeddings(x)
    	
    	dropout_p = 0.1  # 置0概率为0.1
    	max_len = 60  # 句子最大长度
    	
    	# 输入部分的PositionalEncoding类
    	my_pe = PositionalEncoding(d_model, max_len, dropout_p)
    	pe_result = my_pe(embedded_result)
    	
    	# 调用attention函数
    	# 准备q,k,v
    	query = key = value = pe_result  # 自注意力
    	# print('query--->', query.shape)
    	# 准备mask掩码张量 padding_mask
    	# unsqueeze(1) -> 形状(2,1,4) 后续进行masked_fill操作, 会进行广播, 变成 (2,4,4)
    	# 多头注意力机制,需要得到(2,1,1,4)形状mask
    	mask = (x != 0).type(torch.uint8).unsqueeze(1).unsqueeze(2)
    	# casual mask
    	# mask = torch.tril(torch.ones(4, 4))
    	# print('mask--->', mask.shape, mask)
    	# c, p_attn = attention(query, key, value, mask)
    	# print('c--->', c.shape, c)
    	# print('p_attn--->', p_attn.shape, p_attn)
    	
    	head = 8
    	# 创建多头注意力机制类对象
    	my_mha = MultiHeadedAttention(head, d_model)
    	# 调用多头注意力机制对象
    	mha_result = my_mha(query, key, value, mask)
    	print('mha_result--->', mha_result.shape, mha_result)
    	
    	# 创建前馈全连接对象
    	d_ff = 2048
    	my_ff = PositionwiseFeedForward(d_model, d_ff)
    	ff_result = my_ff(mha_result)
    	print('ff_result--->', ff_result.shape)
    ```

### 4.5 规范化层(层归一化)

- 概念

    - 在词的词向量维度进行标准化处理
    - 我 -> [1,2,3,4,5]
    - 计算`我`词维度平均值和标准差, 进行标准化处理, 进行线性计算

- 作用

    - 加速模型训练收敛速度
    - 稳定模型训练/增强模型表达能力

- 层归一化和批归一化区别

    - 层归一化词向量维度进行, 不依赖批次大小, 不存储全局信息
    - 批归一化样本维度进行, 依赖批次大小, 存储全局信息

- 代码实现

    ```python
    # 层归一化层
    class LayerNorn(nn.Module):
    	# todo:1- init方法
    	def __init__(self, features, eps=1e-6):
    		super().__init__()
    		self.eps = eps
    		# 初始化w,全1, w维度数和features维度数一致
    		self.w = nn.Parameter(torch.ones(features))
    		# print('self.w--->', self.w.shape)
    		# 初始化b,全0, b维度数和features维度数一致
    		self.b = nn.Parameter(torch.zeros(features))
    	# todo:2- forward方法
    	def forward(self, x):
    		# 计算x的均值和标准
    		# keepdims: 默认False, 返回二维张量; True, 和原x维度数一致
    		mean = x.mean(dim=-1, keepdims=True)
    		print('mean--->', mean.shape, mean)
    		std = x.std(dim=-1, keepdims=True)
    		# 计算标准化的结果
    		x = self.w * ((x-mean) / (std + self.eps)) + self.b
    		return x
    
    
    if __name__ == '__main__':
    	vocab = 1000  # 词表大小是1000
    	d_model = 512  # 词嵌入维度是512维
    	
    	# 输入x 形状是2 x 4
    	x = torch.LongTensor([[100, 2, 421, 0], [491, 998, 0, 0]])
    	
    	# 输入部分的Embeddings类
    	my_embeddings = Embeddings(vocab, d_model)
    	embedded_result = my_embeddings(x)
    	
    	dropout_p = 0.1  # 置0概率为0.1
    	max_len = 60  # 句子最大长度
    	
    	# 输入部分的PositionalEncoding类
    	my_pe = PositionalEncoding(d_model, max_len, dropout_p)
    	pe_result = my_pe(embedded_result)
    	
    	# 调用attention函数
    	# 准备q,k,v
    	query = key = value = pe_result  # 自注意力
    	# print('query--->', query.shape)
    	# 准备mask掩码张量 padding_mask
    	# unsqueeze(1) -> 形状(2,1,4) 后续进行masked_fill操作, 会进行广播, 变成 (2,4,4)
    	# 多头注意力机制,需要得到(2,1,1,4)形状mask
    	mask = (x != 0).type(torch.uint8).unsqueeze(1).unsqueeze(2)
    	# casual mask
    	# mask = torch.tril(torch.ones(4, 4))
    	# print('mask--->', mask.shape, mask)
    	# c, p_attn = attention(query, key, value, mask)
    	# print('c--->', c.shape, c)
    	# print('p_attn--->', p_attn.shape, p_attn)
    	
    	head = 8
    	# 创建多头注意力机制类对象
    	my_mha = MultiHeadedAttention(head, d_model)
    	# 调用多头注意力机制对象
    	mha_result = my_mha(query, key, value, mask)
    	print('mha_result--->', mha_result.shape, mha_result)
    	
    	# 创建前馈全连接对象
    	d_ff = 2048
    	my_ff = PositionwiseFeedForward(d_model, d_ff)
    	ff_result = my_ff(mha_result)
    	print('ff_result--->', ff_result.shape)
    	
    	# 创建层归一化对象
    	features = d_model
    	eps = 1e-6
    	my_ln = LayerNorn(features, eps)
    	ln_result = my_ln(ff_result)
    	print('ln_result--->', ln_result.shape, ln_result)
    ```

### 4.6 子层连接

- 概念

    - 残差连接: 将输入x和当前层输出相加到一起 
        - 例如: 多头自注意力子层, 将输入x(q/k/v)和子层输出结果加到一起
    - 将残差结果进行norm层归一化操作

- 作用

    - 缓解梯度消失或爆炸
    - 加速模型收敛
    - 训练模型稳定性
    - 保留更多语义信息

- 代码实现

    ```python
    class SublayerConnection(nn.Module):
    	def __init__(self, size, dropout_p=0.1):
    		super().__init__()
    		self.size = size
    		# 创建层归一化对象
    		self.norm = LayerNorm(self.size)
    		self.dropout = nn.Dropout(p=dropout_p)
    	def forward(self, x, sublayer):
    		"""
    		计算子层结果  当前层输入x+dropout(当前层输出)=残差结果->norm
    		:param x: 当前层输入
    		:param sublayer: 当前层对象 方法名/内存地址
    		:return: 子层结果
    		"""
    		x = self.norm(x + self.dropout(sublayer(x)))
    		return x
    		
    	
    
    
    if __name__ == '__main__':
    	vocab = 1000  # 词表大小是1000
    	d_model = 512  # 词嵌入维度是512维
    	
    	# 输入x 形状是2 x 4
    	x = torch.LongTensor([[100, 2, 421, 0], [491, 998, 0, 0]])
    	
    	# 输入部分的Embeddings类
    	my_embeddings = Embeddings(vocab, d_model)
    	embedded_result = my_embeddings(x)
    	
    	dropout_p = 0.1  # 置0概率为0.1
    	max_len = 60  # 句子最大长度
    	
    	# 输入部分的PositionalEncoding类
    	my_pe = PositionalEncoding(d_model, max_len, dropout_p)
    	pe_result = my_pe(embedded_result)
    	
    	# 调用attention函数
    	# 准备q,k,v
    	query = key = value = pe_result  # 自注意力
    	# print('query--->', query.shape)
    	# 准备mask掩码张量 padding_mask
    	# unsqueeze(1) -> 形状(2,1,4) 后续进行masked_fill操作, 会进行广播, 变成 (2,4,4)
    	# 多头注意力机制,需要得到(2,1,1,4)形状mask
    	mask = (x != 0).type(torch.uint8).unsqueeze(1).unsqueeze(2)
    	# casual mask
    	# mask = torch.tril(torch.ones(4, 4))
    	# print('mask--->', mask.shape, mask)
    	# c, p_attn = attention(query, key, value, mask)
    	# print('c--->', c.shape, c)
    	# print('p_attn--->', p_attn.shape, p_attn)
    	
    	head = 8
    	# 创建多头注意力机制类对象
    	my_mha = MultiHeadedAttention(head, d_model)
    	# 定义匿名函数
    	"""
    	def sublayer(x):
    		return my_mha(x, x, x)
    	"""
    	# 变量名 = lambda 参数: 参数的表达式
    	sublayer = lambda x: my_mha(x, x, x)
    	print('sublayer--->', sublayer)
    	
    	# 创建子层连接对象
    	size = d_model
    	my_sc = SublayerConnection(size)
    	# 调用对象
    	sc_result = my_sc(pe_result, sublayer)
    	print('sc_result--->', sc_result.shape, sc_result)
    ```

### 4.7 编码器层

- 概念

    - 由多头自注意力子层和前馈全连接子层组成的结构
    - 等同于ANN中的隐藏层

- 作用

    - 特征提取

- 代码实现

    ```python
    # 创建编码器层
    class EncoderLayer(nn.Module):
    	def __init__(self, size, self_attn, feed_forwad, dropout_p=0.1):
    		super().__init__()
    		self.size = size  # 词向量维度
    		self.self_attn = self_attn  # 多头自注意力机制对象
    		self.feed_forwad = feed_forwad  # 前馈全连接对象
    		self.dropout_p = dropout_p
    		# 克隆两个子层对象
    		# self.sublayer->[对象1, 对象2]
    		self.sublayer = clones(SublayerConnection(self.size, self.dropout_p), 2)
    		print('self.sublayer--->', self.sublayer)
    		
    	def forward(self, x, mask):
    		"""
    		编码器层进行编码
    		:param x: 上一层的输出, 如果是第一层编码器层,x是输入部分的输出
    		:param mask: 填充掩码
    		:return: 编码结果,提取到的新特征向量
    		"""
    		# 多头自主力机制子层计算
    		x = self.sublayer[0](x, lambda x:self.self_attn(x,x,x,mask))
    		# 前馈全连接子层计算
    		x = self.sublayer[1](x, self.feed_forwad)
    		return x
    		
    
    
    if __name__ == '__main__':
    	vocab = 1000  # 词表大小是1000
    	d_model = 512  # 词嵌入维度是512维
    	
    	# 输入x 形状是2 x 4
    	x = torch.LongTensor([[100, 2, 421, 0], [491, 998, 0, 0]])
    	
    	# 输入部分的Embeddings类
    	my_embeddings = Embeddings(vocab, d_model)
    	embedded_result = my_embeddings(x)
    	
    	dropout_p = 0.1  # 置0概率为0.1
    	max_len = 60  # 句子最大长度
    	
    	# 输入部分的PositionalEncoding类
    	my_pe = PositionalEncoding(d_model, max_len, dropout_p)
    	pe_result = my_pe(embedded_result)
    	
    	# 调用attention函数
    	# 准备q,k,v
    	query = key = value = pe_result  # 自注意力
    	# print('query--->', query.shape)
    	# 准备mask掩码张量 padding_mask
    	# unsqueeze(1) -> 形状(2,1,4) 后续进行masked_fill操作, 会进行广播, 变成 (2,4,4)
    	# 多头注意力机制,需要得到(2,1,1,4)形状mask
    	mask = (x != 0).type(torch.uint8).unsqueeze(1).unsqueeze(2)
    	# casual mask
    	# mask = torch.tril(torch.ones(4, 4))
    	# print('mask--->', mask.shape, mask)
    	# c, p_attn = attention(query, key, value, mask)
    	# print('c--->', c.shape, c)
    	# print('p_attn--->', p_attn.shape, p_attn)
    	
    	head = 8
    	# 创建多头注意力机制类对象
    	my_mha = MultiHeadedAttention(head, d_model)
    	# 实例化前馈全连接层对象
    	d_ff=2048
    	my_ff = PositionwiseFeedForward(d_model, d_ff, dropout_p)
    	
    	# 创建编码器层对象
    	my_el = EncoderLayer(d_model, my_mha, my_ff, dropout_p)
    	el_result = my_el(pe_result, mask)
    	print('el_result--->', el_result.shape)
    ```

### 4.8 编码器

- 概念

    - 由多层编码器层组成的结构

- 作用

    - 特征提取

- 代码实现

    ```python
    # 创建编码器
    class Encoder(nn.Module):
    	def __init__(self, layer, N):
    		super().__init__()
    		self.layers = clones(layer, N)
    		# print('self.layers--->', self.layers)
    		# self.norm = LayerNorm(layer.size)
    		
    	def forward(self, x, mask):
    		# 循环遍历编码器, 一层一层计算
    		for layer in self.layers:
    			x = layer(x, mask)
    			
    		return x
    
    
    def test_encoder():
    	vocab = 1000  # 词表大小是1000
    	d_model = 512  # 词嵌入维度是512维
    	
    	# 输入x 形状是2 x 4
    	x = torch.LongTensor([[100, 2, 421, 0], [491, 998, 0, 0]])
    	
    	# 输入部分的Embeddings类
    	my_embeddings = Embeddings(vocab, d_model)
    	embedded_result = my_embeddings(x)
    	
    	dropout_p = 0.1  # 置0概率为0.1
    	max_len = 60  # 句子最大长度
    	
    	# 输入部分的PositionalEncoding类
    	my_pe = PositionalEncoding(d_model, max_len, dropout_p)
    	pe_result = my_pe(embedded_result)
    	
    	# 调用attention函数
    	# 准备q,k,v
    	query = key = value = pe_result  # 自注意力
    	# print('query--->', query.shape)
    	# 准备mask掩码张量 padding_mask
    	# unsqueeze(1) -> 形状(2,1,4) 后续进行masked_fill操作, 会进行广播, 变成 (2,4,4)
    	# 多头注意力机制,需要得到(2,1,1,4)形状mask
    	mask = (x != 0).type(torch.uint8).unsqueeze(1).unsqueeze(2)
    	# casual mask
    	# mask = torch.tril(torch.ones(4, 4))
    	# print('mask--->', mask.shape, mask)
    	# c, p_attn = attention(query, key, value, mask)
    	# print('c--->', c.shape, c)
    	# print('p_attn--->', p_attn.shape, p_attn)
    	
    	head = 8
    	# 创建多头注意力机制类对象
    	my_mha = MultiHeadedAttention(head, d_model)
    	# 实例化前馈全连接层对象
    	d_ff = 2048
    	my_ff = PositionwiseFeedForward(d_model, d_ff, dropout_p)
    	
    	# 创建深拷贝对象
    	c = copy.deepcopy
    	# 创建编码器层对象
    	# 深拷贝对象, 每个对象的内存地址不同, 不共享参数
    	my_el = EncoderLayer(d_model, c(my_mha), c(my_ff), dropout_p)
    	
    	# 创建编码器对象
    	my_encoder = Encoder(my_el, 2)
    	encoder_result = my_encoder(pe_result, mask)
    	# print('encoder_result--->', encoder_result.shape)
    	return encoder_result
    
    if __name__ == '__main__':
    	encoder_result = test_encoder()
    	print('encoder_result--->', encoder_result.shape)
    ```

### 4.9 编码器结构完整代码

```python
from input import *
import copy


# 定义缩放点积注意力规则函数, 方便后续调用
def attention(query, key, value, mask=None, dropout=None):
	"""
	注意力计算封装函数
	:param query: 输入x  解码器掩码多头自注意力子层输出
	:param key: 输入x  编码器输出结果
	:param value: 输入x  编码器输出结果
	:param mask: 是否掩码
	:param dropout: dropout层对象 函数名
	:return: 动态c, 权重概率矩阵
	"""
	# todo:1- 获取d_k, 词维度数
	d_k = query.shape[-1]
	# print('d_k--->', d_k)
	# todo:2- q和k计算权重分数矩阵
	scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
	# print('scores--->', scores.shape, scores)
	# todo:3- 判断是否需要进行掩码操作
	if mask is not None:
		scores = scores.masked_fill(mask == 0, value=-1e9)
		# print('=' * 50)
	# print('scores--->', scores.shape, scores)
	# todo:4- 权重分数矩阵进行softmax操作, 得到权重概率矩阵
	p_attn = torch.softmax(scores, dim=-1)
	# print('p_attn--->', p_attn.shape, p_attn)
	# todo:5- 判断是否对权重概率矩阵进行dropout正则化
	if dropout is not None:
		p_attn = dropout(p_attn)
	# todo:6- 计算动态c矩阵
	c = torch.matmul(p_attn, value)
	# print('c--->', c.shape, c)
	return c, p_attn


# 定义克隆函数, 用于克隆不同子层
def clones(module, N):
	return nn.ModuleList(copy.deepcopy(module) for _ in range(N))


# 创建多头注意力机制类
class MultiHeadedAttention(nn.Module):
	# todo:1- init方法
	def __init__(self, head, d_model, dropout_p=0.1):
		super().__init__()
		assert d_model % head == 0, 'd_model不能被head整数'
		self.d_k = d_model // head
		self.dropout = nn.Dropout(p=dropout_p)
		self.head = head
		# 初始为None, 还没有计算注意力
		self.attn = None
		# 4个线性层
		# 前3个分别对q,k,v进行线性学习
		# 第1个对多头注意力拼接结果进行线性学习
		self.linears = clones(nn.Linear(d_model, d_model), 4)
	
	# print('self.linears--->', self.linears)
	# todo:2- forward方法
	def forward(self, query, key, value, mask=None):
		# todo:1- 获取batch_size大小
		batch_size = query.size()[0]
		# print('batch_size--->', batch_size)
		# todo:2- 准备空列表, 存储线性计算+变形结果
		output_list = []
		# todo:3- q,k,v分别进行线性计算
		for model, x in zip(self.linears, (query, key, value)):
			# print('model--->', model)
			# print('x--->', x)
			output = model(x)
			# todo:4- 线性计算结果变形 -> (batch_size, seq_len, head, d_k)
			# transpose(1, 2):词数和词向量相邻, 更好的学习特征
			output = output.view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
			# todo:5- 将变形结果保存到空列表中
			output_list.append(output)
		# 获取q, k, v
		# print('output_list--->', len(output_list))
		query = output_list[0]
		key = output_list[1]
		value = output_list[2]
		# todo:6- 计算多头注意力, 调用attention函数  (batch_size, seq_len, head, d_k)
		x, p_attn = attention(query, key, value, mask)
		# print('x--->', x.shape)
		# todo:7- 多头注意力结果变形 -> (batch_size, seq_len, word_dim)
		x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.d_k)
		# print('x--->', x.shape)
		# todo:8- 经过线性层计算返回输出结果
		# self.linears[-1]: 线性层对象
		x = self.linears[-1](x)
		return x


# 前馈全连接层
class PositionwiseFeedForward(nn.Module):
	def __init__(self, d_model, d_ff, dropout_p=0.1):
		super().__init__()
		# 定义两层线性层
		# d_ff>d_model
		self.linear1 = nn.Linear(d_model, d_ff)
		self.linear2 = nn.Linear(d_ff, d_model)
		# 定义dropout层
		self.dropout = nn.Dropout(p=dropout_p)
	
	def forward(self, x):
		output = torch.relu(self.linear1(x))
		output = self.dropout(output)
		# print('output--->', output.shape)
		return self.linear2(output)
	
# 层归一化层
class LayerNorm(nn.Module):
	# todo:1- init方法
	def __init__(self, features, eps=1e-6):
		super().__init__()
		self.eps = eps
		# 初始化w,全1, w维度数和features维度数一致
		self.w = nn.Parameter(torch.ones(features))
		# print('self.w--->', self.w.shape)
		# 初始化b,全0, b维度数和features维度数一致
		self.b = nn.Parameter(torch.zeros(features))
	# todo:2- forward方法
	def forward(self, x):
		# 计算x的均值和标准
		# keepdims: 默认False, 返回二维张量; True, 和原x维度数一致
		mean = x.mean(dim=-1, keepdims=True)
		# print('mean--->', mean.shape, mean)
		std = x.std(dim=-1, keepdims=True)
		# 计算标准化的结果
		x = self.w * ((x-mean) / (std + self.eps)) + self.b
		return x
	
class SublayerConnection(nn.Module):
	def __init__(self, size, dropout_p=0.1):
		super().__init__()
		self.size = size
		# 创建层归一化对象
		self.norm = LayerNorm(self.size)
		self.dropout = nn.Dropout(p=dropout_p)
	def forward(self, x, sublayer):
		"""
		计算子层结果  当前层输入x+dropout(当前层输出)=残差结果->norm
		:param x: 当前层输入
		:param sublayer: 当前层对象 方法名/内存地址
		:return: 子层结果
		"""
		x = self.norm(x + self.dropout(sublayer(x)))
		return x
	
	
# 创建编码器层
class EncoderLayer(nn.Module):
	def __init__(self, size, self_attn, feed_forwad, dropout_p=0.1):
		super().__init__()
		self.size = size  # 词向量维度
		self.self_attn = self_attn  # 多头自注意力机制对象
		self.feed_forwad = feed_forwad  # 前馈全连接对象
		self.dropout_p = dropout_p
		# 克隆两个子层对象
		# self.sublayer->[对象1, 对象2]
		self.sublayer = clones(SublayerConnection(self.size, self.dropout_p), 2)
		# print('self.sublayer--->', self.sublayer)
		
	def forward(self, x, mask):
		"""
		编码器层进行编码
		:param x: 上一层的输出, 如果是第一层编码器层,x是输入部分的输出
		:param mask: 填充掩码
		:return: 编码结果,提取到的新特征向量
		"""
		# 多头自主力机制子层计算
		x = self.sublayer[0](x, lambda x:self.self_attn(x,x,x,mask))
		# 前馈全连接子层计算
		x = self.sublayer[1](x, self.feed_forwad)
		return x
		
		
# 创建编码器
class Encoder(nn.Module):
	def __init__(self, layer, N):
		super().__init__()
		self.layers = clones(layer, N)
		# print('self.layers--->', self.layers)
		# self.norm = LayerNorm(layer.size)
		
	def forward(self, x, mask):
		# 循环遍历编码器, 一层一层计算
		for layer in self.layers:
			x = layer(x, mask)
			
		return x


def test_encoder():
	vocab = 1000  # 词表大小是1000
	d_model = 512  # 词嵌入维度是512维
	
	# 输入x 形状是2 x 4
	x = torch.LongTensor([[100, 2, 421, 0], [491, 998, 0, 0]])
	
	# 输入部分的Embeddings类
	my_embeddings = Embeddings(vocab, d_model)
	embedded_result = my_embeddings(x)
	
	dropout_p = 0.1  # 置0概率为0.1
	max_len = 60  # 句子最大长度
	
	# 输入部分的PositionalEncoding类
	my_pe = PositionalEncoding(d_model, max_len, dropout_p)
	pe_result = my_pe(embedded_result)
	
	# 调用attention函数
	# 准备q,k,v
	query = key = value = pe_result  # 自注意力
	# print('query--->', query.shape)
	# 准备mask掩码张量 padding_mask
	# unsqueeze(1) -> 形状(2,1,4) 后续进行masked_fill操作, 会进行广播, 变成 (2,4,4)
	# 多头注意力机制,需要得到(2,1,1,4)形状mask
	mask = (x != 0).type(torch.uint8).unsqueeze(1).unsqueeze(2)
	# casual mask
	# mask = torch.tril(torch.ones(4, 4))
	# print('mask--->', mask.shape, mask)
	# c, p_attn = attention(query, key, value, mask)
	# print('c--->', c.shape, c)
	# print('p_attn--->', p_attn.shape, p_attn)
	
	head = 8
	# 创建多头注意力机制类对象
	my_mha = MultiHeadedAttention(head, d_model)
	# 实例化前馈全连接层对象
	d_ff = 2048
	my_ff = PositionwiseFeedForward(d_model, d_ff, dropout_p)
	
	# 创建深拷贝对象
	c = copy.deepcopy
	# 创建编码器层对象
	# 深拷贝对象, 每个对象的内存地址不同, 不共享参数
	my_el = EncoderLayer(d_model, c(my_mha), c(my_ff), dropout_p)
	
	# 创建编码器对象
	my_encoder = Encoder(my_el, 2)
	encoder_result = my_encoder(pe_result, mask)
	# print('encoder_result--->', encoder_result.shape)
	return encoder_result

if __name__ == '__main__':
	encoder_result = test_encoder()
	print('encoder_result--->', encoder_result.shape)
```

## 5 解码器

### 5.1 解码器层

- 概念

    - 由掩码多头自注意力子层、编码器-解码器多头一般注意力子层和前馈全连接子层组成的结构

- 作用

    - 提取预测结果上文的向量表示

- 代码实现

    ```python
    from encoder import *
    
    
    # 创建解码器层类
    class DecoderLayer(nn.Module):
    	def __init__(self, size, self_attn, src_attn, feed_forward, dropout_p=0.1 ):
    		super().__init__()
    		self.size = size
    		self.self_attn = self_attn
    		self.src_attn = src_attn
    		self.feed_forward = feed_forward
    		# 初始化子层对象
    		self.sublayers = clones(SublayerConnection(self.size, dropout_p), 3)
    		
    	def forward(self, x, m, padding_mask, casual_mask):
    		"""
    		
    		:param x: 解码器的输入
    		:param m: 编码器的输出
    		:param padding_mask: 填充掩码
    		:param casual_mask: 因果掩码
    		:return:
    		"""
    		# 掩码多头自注意力子层
    		x = self.sublayers[0](x, lambda x: self.self_attn(x,x,x,casual_mask))
    		# 掩码一般注意力子层
    		x = self.sublayers[1](x, lambda x: self.self_attn(x,m,m,padding_mask))
    		# 前馈全连接子层
    		x = self.sublayers[2](x, self.feed_forward)
    		return x
    	
    if __name__ == '__main__':
    	vocab = 1000  # 词表大小是1000
    	d_model = 512
    	
    	# 输入x 形状是2 x 4
    	x = torch.LongTensor([[100, 2, 421, 508], [491, 998, 1, 221]])
    	
    	my_embeddings = Embeddings(vocab, d_model)
    	embedded_result = my_embeddings(x)  # [2, 4, 512]
    	
    	dropout_p = 0.2  # 置0概率为0.2
    	max_len = 60  # 句子最大长度
    	my_pe = PositionalEncoding(d_model, max_len, dropout_p)
    	pe_result = my_pe(embedded_result)
    	
    	# 类的实例化参数与解码器层类似, 相比多出了src_attn, 但是和self_attn是同一个类.
    	head = 8
    	d_ff = 64
    	size = 512
    	self_attn = src_attn = MultiHeadedAttention(head, d_model, dropout_p)
    	
    	# 前馈全连接层也和之前相同
    	my_ff = PositionwiseFeedForward(d_model, d_ff, dropout_p)
    	
    	# 产生编码器结果
    	# 注意此函数返回编码以后的结果 要有返回值, dm_test_Encoder函数后return en_result
    	en_result = test_encoder()
    	# 因果掩码
    	casual_mask = torch.tril(torch.ones(size=(1, 1, 4, 4))).type(torch.uint8)
    	# 填充掩码
    	padding_mask = (x != 0).type(torch.uint8).unsqueeze(1).unsqueeze(2)
    	
    	# 创建解码器层对象
    	my_dl = DecoderLayer(size, self_attn, src_attn, my_ff, dropout_p)
    	dl_result = my_dl(pe_result, en_result, padding_mask, casual_mask)
    	print('dl_result--->', dl_result.shape)
    ```

### 5.2 解码器

- 概念

    - 由N层解码器层组成的结构, transformer中默认6层

- 作用

    - 提取预测结果上文的向量表示

- 代码实现

    ```python
    # 创建编码器类
    class Decoder(nn.Module):
    	def __init__(self, layer, N):
    		super().__init__()
    		self.layers = clones(layer, N)
    		
    	def forward(self, x, m, padding_mask, casual_mask):
    		for layer in self.layers:
    			x = layer(x, m, padding_mask, casual_mask)
    		return x
    	
    if __name__ == '__main__':
    	vocab = 1000  # 词表大小是1000
    	d_model = 512
    	
    	# 输入x 形状是2 x 4
    	x = torch.LongTensor([[100, 2, 421, 508], [491, 998, 1, 221]])
    	
    	my_embeddings = Embeddings(vocab, d_model)
    	embedded_result = my_embeddings(x)  # [2, 4, 512]
    	
    	dropout_p = 0.2  # 置0概率为0.2
    	max_len = 60  # 句子最大长度
    	my_pe = PositionalEncoding(d_model, max_len, dropout_p)
    	pe_result = my_pe(embedded_result)
    	
    	# 类的实例化参数与解码器层类似, 相比多出了src_attn, 但是和self_attn是同一个类.
    	head = 8
    	d_ff = 64
    	size = 512
    	self_attn = src_attn = MultiHeadedAttention(head, d_model, dropout_p)
    	
    	# 前馈全连接层也和之前相同
    	my_ff = PositionwiseFeedForward(d_model, d_ff, dropout_p)
    	
    	# 产生编码器结果
    	# 注意此函数返回编码以后的结果 要有返回值, dm_test_Encoder函数后return en_result
    	en_result = test_encoder()
    	# 因果掩码
    	casual_mask = torch.tril(torch.ones(size=(1, 1, 4, 4))).type(torch.uint8)
    	# 填充掩码
    	padding_mask = (x != 0).type(torch.uint8).unsqueeze(1).unsqueeze(2)
    	
    	c = copy.deepcopy
    	# 创建解码器层对象
    	my_dl = DecoderLayer(size, c(self_attn), c(src_attn), c(my_ff), dropout_p)
    	
    	# 创建解码器对象
    	my_decoder = Decoder(my_dl, 2)
    	de_result = my_decoder(pe_result, en_result, padding_mask, casual_mask)
    	print('de_result--->', de_result.shape, de_result)
    ```

### 5.3 解码器部分完整代码

```python
import copy

from encoder import *


# 创建解码器层类
class DecoderLayer(nn.Module):
	def __init__(self, size, self_attn, src_attn, feed_forward, dropout_p=0.1 ):
		super().__init__()
		self.size = size
		self.self_attn = self_attn
		self.src_attn = src_attn
		self.feed_forward = feed_forward
		# 初始化子层对象
		self.sublayers = clones(SublayerConnection(self.size, dropout_p), 3)
		
	def forward(self, x, m, padding_mask, casual_mask):
		"""
		:param x: 解码器的输入
		:param m: 编码器的输出
		:param padding_mask: 填充掩码
		:param casual_mask: 因果掩码
		:return:
		"""
		# 掩码多头自注意力子层
		x = self.sublayers[0](x, lambda x: self.self_attn(x,x,x,casual_mask))
		# 掩码一般注意力子层
		x = self.sublayers[1](x, lambda x: self.self_attn(x,m,m,padding_mask))
		# 前馈全连接子层
		x = self.sublayers[2](x, self.feed_forward)
		return x
	
# 创建编码器类
class Decoder(nn.Module):
	def __init__(self, layer, N):
		super().__init__()
		self.layers = clones(layer, N)
		
	def forward(self, x, m, padding_mask, casual_mask):
		for layer in self.layers:
			x = layer(x, m, padding_mask, casual_mask)
		return x
	
if __name__ == '__main__':
	vocab = 1000  # 词表大小是1000
	d_model = 512
	
	# 输入x 形状是2 x 4
	x = torch.LongTensor([[100, 2, 421, 508], [491, 998, 1, 221]])
	
	my_embeddings = Embeddings(vocab, d_model)
	embedded_result = my_embeddings(x)  # [2, 4, 512]
	
	dropout_p = 0.2  # 置0概率为0.2
	max_len = 60  # 句子最大长度
	my_pe = PositionalEncoding(d_model, max_len, dropout_p)
	pe_result = my_pe(embedded_result)
	
	# 类的实例化参数与解码器层类似, 相比多出了src_attn, 但是和self_attn是同一个类.
	head = 8
	d_ff = 64
	size = 512
	self_attn = src_attn = MultiHeadedAttention(head, d_model, dropout_p)
	
	# 前馈全连接层也和之前相同
	my_ff = PositionwiseFeedForward(d_model, d_ff, dropout_p)
	
	# 产生编码器结果
	# 注意此函数返回编码以后的结果 要有返回值, dm_test_Encoder函数后return en_result
	en_result = test_encoder()
	# 因果掩码
	casual_mask = torch.tril(torch.ones(size=(1, 1, 4, 4))).type(torch.uint8)
	# 填充掩码
	padding_mask = (x != 0).type(torch.uint8).unsqueeze(1).unsqueeze(2)
	
	c = copy.deepcopy
	# 创建解码器层对象
	my_dl = DecoderLayer(size, c(self_attn), c(src_attn), c(my_ff), dropout_p)
	
	# 创建解码器对象
	my_decoder = Decoder(my_dl, 2)
	de_result = my_decoder(pe_result, en_result, padding_mask, casual_mask)
	print('de_result--->', de_result.shape, de_result)
```

## 6 输出

- 概念

    - 由1个线性层和1个softmax层组成的结构
- 作用

    - 将解码器输出的向量表示转换成概率表示

- 代码实现

    ```python
    from decoder import *
    
    
    # 创建全连接层类
    class Generator(nn.Module):
    	def __init__(self, d_model, vocab_size):
    		super().__init__()
    		# 输出维度=词表大小
    		self.out = nn.Linear(d_model, vocab_size)
    	
    	def forward(self, x):
    		return torch.log_softmax(self.out(x), dim=-1)
    
    
    if __name__ == '__main__':
    	# 获取解码器结果
    	dl_result = test_decoder()
    	d_model = 512
    	vocab_size = 10000
    	# 创建输出对象
    	my_generator = Generator(d_model, vocab_size)
    	output = my_generator(dl_result)
    	print(output.shape, output)
    ```

## 7 transformer模型搭建

```python
from output import *


# 创建编码器解码器对象
class EncoderDecoder(nn.Module):
	def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
		super().__init__()
		# 初始化各层对象
		self.encoder = encoder
		self.decoder = decoder
		# src_embed: 编码器输入层对象 由词嵌入对象+位置编码对象组成
		# 后续代码中使用 nn.Sequential()类将词嵌入对象+位置编码对象顺序合并到一起, 顺序执行
		self.src_embed = src_embed
		# tgt_embed: 解码器输入层对象 由词嵌入对象+位置编码对象组成
		self.tgt_embed = tgt_embed
		self.generator = generator
	
	def forward(self, src, tgt, src_mask, tgt_mask):
		# 编码器编码, 得到语义张量c
		m = self.encoder(self.src_embed(src), src_mask)
		# 解码器解码
		x = self.decoder(self.src_embed(tgt), m, src_mask, tgt_mask)
		output = self.generator(x)
		return output


# 创建transformer模型
def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout_p=0.1):
	c = copy.deepcopy
	# 创建前馈全连接层
	ff = PositionwiseFeedForward(d_model, d_ff, dropout_p)
	# 创建位置编码对象
	pe = PositionalEncoding(d_model=d_model, dropout_p=dropout_p)
	# 创建多头注意力对象
	attn = MultiHeadedAttention(h, d_model, dropout_p)
	# 创建编码器层对象
	el = EncoderLayer(d_model, c(attn), c(ff), dropout_p)
	# 创建解码器层对象
	dl = DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout_p)
	# 创建模型对象
	model = EncoderDecoder(
		Encoder(el, N),
		Decoder(dl, N),
		# 创建容器列表, 后续执行按照容器中的对象顺序执行
		nn.Sequential(Embeddings(src_vocab, d_model), c(pe)),
		nn.Sequential(Embeddings(tgt_vocab, d_model), c(pe)),
		Generator(d_model, tgt_vocab)
	)
	
	# 将模型对象参数进行初始化
	for p in model.parameters():
		# 判断w和b的维度是否大于1维
		if p.dim() > 1:
			nn.init.xavier_uniform_(p)
	return model


if __name__ == '__main__':
	source_vocab = 512
	target_vocab = 512
	# 创建transformer模型对象
	model = make_model(source_vocab, target_vocab)
	print('model--->', model)
	# 获取模型的encoder部分
	print('model.encoder--->', model.encoder)
	# 获取模型encoder部分第1层子层
	print('model.encoder.layers[0]--->', model.encoder.layers[0])
	
	# 假设源数据与目标数据相同, 实际中并不相同
	source = target = torch.LongTensor([[1, 2, 3, 8], [3, 4, 1, 8]])
	
	# 假设src_mask与tgt_mask相同，实际中并不相同
	source_mask = target_mask = torch.tril(torch.ones(size=(8, 4, 4))).type(torch.uint8)
	
	# 调用模型得到预测结果
	output = model(source, target, source_mask, target_mask)
	print('output--->', output.shape, output)
```

## 8 fasttext工具

### 8.1 介绍

- 概念
- 作用
- 特点

### 8.2 架构(了解)

- fasttext模型组成
- 层次softmax
- 负采样 

### 8.3 文本分类

- 概念
- 代码实现

### 8.4 训练词向量



### 8.5 词向量迁移

## 9 迁移学习

- 概念
- 预训练模型 pre-model
- 微调 fine-tuning































