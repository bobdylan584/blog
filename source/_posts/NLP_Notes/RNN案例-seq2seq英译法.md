---
title: RNN案例-seq2seq英译法
date: 2023-12-20 06:29:52
tag: NLP_Notes
---

# RNN案例-seq2seq英译法

## 1 RNN案例-seq2seq英译法

### 1.1 seq2seq模型介绍

- 模型结构
  - 编码器 encoder
  - 解码器 decoder
  - 编码器和解码器中可以使用RNN模型或者是transformer模型
- 工作流程
  - 编码器生成上下文语义张量 -> 什么是nlp? 将问题转换成语义张量 
  - 解码器根据编码器的语义张量和上一时间步的预测值以及上一时间步的隐藏状态值进行当前时间步的预测
    - 自回归模式
- 局限性
  - 信息瓶颈问题
  - 长序列问题

### 1.2 数据集介绍

![1748849163083](1748849163083.png)

- 每行样本由英文句子和法文句子对组成, 中间用`\t`分隔开
- 英文句子是编码器的输入序列, 法文句子是解码器的输出序列(预测序列)对应的真实序列

### 1.3 案例实现步骤

#### 1.3.1 文本清洗工具函数

- `utils.py`

  ```python
  # 用于正则表达式
  import re
  # 用于构建网络结构和函数的torch工具包
  import torch
  import torch.nn as nn
  from torch.utils.data import Dataset, DataLoader
  # torch中预定义的优化方法工具包
  import torch.optim as optim
  import time
  # 用于随机生成数据
  import random
  import numpy as np
  import matplotlib.pyplot as plt
  
  # 定义变量
  # 选择设备 cpu/gpu
  # 'cuda'->使用所有显卡  'cuda:0'->使用第一张显卡
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  # 起始符号下标
  # sos -> start of sentences
  SOS_token = 0
  # 结束符号下标
  EOS_token = 1
  # 文件路径
  data_path = 'data/eng-fra-v2.txt'
  # 最大句子长度, 预处理分析的结果
  MAX_LENGTH = 10
  
  
  # 定义处理文本的工具函数  处理句子中的特殊符号/大小写/换行符
  def normalizeString(s: str):
  	# 转换成小写, 并删掉两端的空白符号
  	str = s.lower().strip()
  	# 正则表达式匹配标签符号'.?!' 转换成 ' .?!'
  	str = re.sub(r'([.!?])', r' \1', str)
  	# print('str--->', str)
  	# 正则表达式匹配除a-z.!?之外的其他的符号 转换成 ' '
  	str = re.sub(r'[^a-z.!?]+', r' ', str)
  	# print('str--->', str)
  	return str
  
  
  if __name__ == '__main__':
  	str1 = 'I m sad.@'
  	normalizeString(str1)
  ```

#### 1.3.2 数据预处理

- `preprocess.py`

  - 清洗文本和构建词表

    ```python
    from utils import *
    
    
    def my_getdata():
    	# todo:1- 读取文件数据集, 得到 [[英文句子1, 法文句子1], [英文句子2, 法文句子2], ...]内存数据集
    	# 1-1 with open 读取文件数据集
    	with open(data_path, 'r', encoding='utf-8') as f:
    		my_lines = f.read().strip().split('\n')
    		# print('my_lines --->', my_lines)
    	# 1-2 获取 [[英文句子1, 法文句子1], [英文句子2, 法文句子2], ...] 数据集格式
    	# 定义两个空列表
    	tmp_pair, my_pairs = [], []
    	# 循环遍历my_lines
    	for line in my_lines:
    		# print('line--->', line)  # i m .	j ai ans .
    		# 对my_lines中每行样本使用\t分割符进行分割后再循环遍历
    		for item in line.split('\t'):
    			# print('item--->', item)
    			# 将每行样本中的英文句子和法文句子使用工具函数进行清洗, 保存到tmp_pair列表中
    			tmp_pair.append(normalizeString(item))
    		# 将tmp_pair列表保存到my_pairs列表中
    		my_pairs.append(tmp_pair)
    		# 重置tmp_pair列表
    		tmp_pair = []
    	# print('my_pairs的长度为--->', len(my_pairs))
    	# print('my_pairs[:4]--->', my_pairs[:4])
    	
    	# todo:2-构建英文和法文词表 {词:下标} {下标:词}
    	# 2-0: 初始化词表, 有SOS和EOS两个词
    	english_word2index = {'SOS':0, 'EOS':1}
    	# 定义第3个词起始下标
    	english_word_n = 2
    	french_word2index = {'SOS': 0, 'EOS': 1}
    	french_word_n = 2
    	
    	# 2-1: 循环遍历my_pairs [['i m .', 'j ai ans .'], ...]
    	for pair in my_pairs:
    		# print('pair--->', pair)  # ['i m .', 'j ai ans .']
    		# 2-2: 对英文句子或法文句子根据 ' '空格进行分割, 再进行循环遍历
    		for word in pair[0].split(' '):
    			# print('word--->', word)  # i  m  .
    			# 2-3: 使用if语句, 判断当前词是否在词表中, 如果不在添加进去
    			if word not in english_word2index.keys():
    				english_word2index[word] = english_word_n
    				# 更新词下标
    				english_word_n+=1
    		for word in pair[1].split(' '):
    			# 2-3: 使用if语句, 判断当前词是否在词表中, 如果不在添加进去
    			if word not in french_word2index.keys():
    				french_word2index[word] = french_word_n
    				# 更新词下标
    				french_word_n+=1
    	
    	# 2-4 获取{下标:词}格式词表
    	english_index2word = {v:k for k, v in english_word2index.items()}
    	french_index2word = {v:k for k, v in french_word2index.items()}
    	# print('english_word2index--->', len(english_word2index), english_word2index)
    	# print('french_word2index--->', len(french_word2index), french_word2index)
    	# print('english_index2word--->', len(english_index2word), english_index2word)
    	# print('french_index2word--->', len(french_index2word), french_index2word)
    	# print('english_word_n--->', english_word_n)
    	# print('french_word_n--->', french_word_n)
    	return english_word2index, english_index2word, english_word_n, french_word2index, french_index2word, french_word_n, my_pairs
    
    
    if __name__ == '__main__':
    	(english_word2index, english_index2word, english_word_n, 
    	 french_word2index, french_index2word, french_word_n, my_pairs) = my_getdata()
    ```

  - 构建数据源对象

    ```python
    # 自定义张量数据源类
    class MyPairsDataset(Dataset):
    	# todo:1- init构造方法, 初始化属性
    	def __init__(self, my_pairs, english_word2index, french_word2index):
    		self.my_pairs = my_pairs  # [[], [], ...]
    		self.english_word2index = english_word2index
    		self.french_index2word = french_word2index
    		# 获取数据集长度
    		self.sample_len = len(my_pairs)
    	
    	# todo:2- len方法, 返回数据集的长度
    	def __len__(self):
    		return self.sample_len
    	
    	# todo:3- getitem方法, 对数据进行处理, 转换成张量数据对象
    	def __getitem__(self, index):
    		"""
    		转换成张量数据对象
    		:param index: 数据集的下标 -> 第index个样本
    		:return: tensor_x, tensor_y
    		"""
    		# 3-1: 修正index, 防止超过下标边界
    		index = min(max(index, 0), self.sample_len - 1)
    		# print('index--->', index)
    		# 3-2: 获取当前index样本中的 x和y
    		x = self.my_pairs[index][0]
    		y = self.my_pairs[index][1]
    		# print('x--->', x)
    		# print('y--->', y)
    		# 3-3: 将x和y的字符串数据转换成下标表示  词表
    		# self.english_word2index[word]: 根据key获取字典中的value
    		x = [self.english_word2index[word] for word in x.split(' ')]
    		y = [self.french_index2word[word] for word in y.split(' ')]
    		# print('x--->', x)
    		# print('y--->', y)
    		# 3-4: 每个样本最后加EOS下标 结束符号
    		x.append(EOS_token)
    		y.append(EOS_token)
    		# print('x--->', x)
    		# print('y--->', y)
    		# 3-5: 将下标列表转换成张量对象
    		# device: 将张量创建到对应的设备上 GPU/CPU
    		tensor_x = torch.tensor(x, dtype=torch.long, device=device)
    		tensor_y = torch.tensor(y, dtype=torch.long, device=device)
    		# print('tensor_x--->', tensor_x)
    		# print('tensor_y--->', tensor_y)
    		return tensor_x, tensor_y
    
    
    if __name__ == '__main__':
    	(english_word2index, english_index2word, english_word_n,
    	 french_word2index, french_index2word, french_word_n, my_pairs) = my_getdata()
    	# 创建自定义数据源对象
    	my_dataset = MyPairsDataset(my_pairs, english_word2index, french_word2index)
    	print('my_dataset数据集条目数--->', len(my_dataset))
    	print(my_dataset[0])
    	# 创建数据加载器对象
    	my_dataloader = DataLoader(dataset=my_dataset, batch_size=1, shuffle=True)
    	# 循环遍历数据加载器
    	for i, (x, y) in enumerate(my_dataloader):
    		print('x--->', x.shape, x)
    		print('y--->', y.shape, y)
    		break
    ```

#### 1.3.3 构建基于GRU的编码器和解码器

- 构建基于GRU的编码器

  ```python
  from preprocess import *
  class EncoderRNN(nn.Module):
  	# todo:1- 定义构造方法 init
  	def __init__(self, input_size, hidden_size):
  		super().__init__()
  		# 输入特征维度属性  input_size是英文词表的大小
  		self.input_size = input_size
  		# 词嵌入层和隐藏层特征维度属性  共用
  		self.hidden_size = hidden_size
  		# 词嵌入层对象属性
  		self.embedding = nn.Embedding(num_embeddings=self.input_size,
  		                              embedding_dim=self.hidden_size)
  		# gru层对象属性
  		# input_size: 上一层输出特征维度数
  		# hidden_size: 当前层输出特征维度数
  		# batch_first: x和hidden形状 -> (句子数, 句子长度, 词维度)
  		self.gru = nn.GRU(input_size=self.hidden_size, hidden_size=self.hidden_size, batch_first=True)
  	# todo:2- 定义前向传播方法 forward
  	def forward(self, input, hidden):
  		# print('input--->', input.shape)
  		# 词嵌入操作 词向量化
  		embedded = self.embedding(input)
  		# print('embedded--->', embedded.shape)
  		# gru层前向传播操作
  		output, hn = self.gru(embedded, hidden)
  		# print('output--->', output.shape)
  		# print('hn--->', hn.shape)
  		return output, hn
  	# todo:3- 定义初始化隐藏状态值方法 inithidden
  	def inithidden(self):
  		return torch.zeros(size=(1, 1, self.hidden_size), device=device)
  
  
  if __name__ == '__main__':
  	# 获取数据
  	(english_word2index, english_index2word, english_word_n,
  	 french_word2index, french_index2word, french_word_n, my_pairs) = my_getdata()
  	# 创建张量数据集
  	my_dataset = MyPairsDataset(my_pairs, english_word2index, french_word2index)
  	# 创建数据加载器
  	# batch_size: 当前设置为1, 因为句子长度不一致
  	my_dataloader = DataLoader(dataset=my_dataset, batch_size=1, shuffle=True)
  	# 创建编码器对象
  	my_encoderrnn = EncoderRNN(input_size=english_word_n, hidden_size=256).to(device=device)
  	for i, (x, y) in enumerate(my_dataloader):
  		# 一次性喂数据
  		# 初始化隐藏状态值
  		hidden = my_encoderrnn.inithidden()
  		encoder_output, hn = my_encoderrnn(x, hidden)
  		print('encoder_output--->', encoder_output.shape)
  		print('hn--->', hn.shape)
  		
  		# 一个时间步一个时间步喂数据, gru底层实现  了解,解码器需要这样操作
  		hidden = my_encoderrnn.inithidden()
  		# x.shape[1]: 获取当前x的token数, 时间步数
  		for j in range(x.shape[1]):
  			# print('x--->', x)
  			# print('x[0]--->', x[0])
  			# print('x[0][j]--->', x[0][j])
  			tmp_x = x[0][j].view(1, -1)
  			print('tmp_x--->', tmp_x)
  			output, hidden = my_encoderrnn(tmp_x, hidden)
  		print('观察：最后一个时间步output输出是否相等')  # hidden_size = 8 效果比较好
  		print('encoder_output[0][-1]===>', encoder_output[0][-1])
  		print('output===>', output)
  		break
  ```

- 构建基于GRU的解码器

  ```python
  from encoderrnn import *
  
  class DecoderRNN(nn.Module):
  	# todo:1- 定义构造方法 init
  	def __init__(self, output_size, hidden_size):
  		super().__init__()
  		# 初始化法文词表大小维度属性=线性输出层的维度
  		self.output_size = output_size
  		# 初始化gru隐藏层和词嵌入层的维度属性  共用
  		self.hidden_size = hidden_size
  		# 初始化词嵌入层
  		# num_embeddings: 法文词表大小
  		# embedding_dim: 词向量初始维度
  		self.embeding = nn.Embedding(num_embeddings=self.output_size, embedding_dim=self.hidden_size)
  		# 初始化gru层
  		self.gru = nn.GRU(input_size=self.hidden_size, hidden_size=self.hidden_size, batch_first=True)
  		
  		# 初始化全连接层 线性层+激活层
  		# out_features: 法文词表大小  预测出n个词的生成概率
  		self.out = nn.Linear(in_features=self.hidden_size, out_features=self.output_size)
  		# dim:一定是-1, 按行处理
  		self.softmax = nn.LogSoftmax(dim=-1)
  
  	# todo:2- 定义前向传播方法 forward
  	def forward(self, input, hidden):
  		print('input--->', input.shape)
  		# 词嵌入操作
  		embedded = self.embeding(input)
  		print('embedded--->', embedded.shape)
  		# 通过relu激活函数引入非线性因素, 防止过拟合(x<0置为0, 神经元死亡)
  		embedded = torch.relu(embedded)
  		print('embedded--->', embedded.shape)
  		# gru层操作
  		# ouput: 输入input的语义信息, 形状为(句子数, 句子长度, 词维度) 三维
  		output, hidden = self.gru(embedded, hidden)
  		print('output--->', output.shape, output)
  		# 全连接层操作
  		# output[0]: 全连接层一般是二维数据, 所以要取出当前token的二维表示
  		# 返回的output是 logsoftmax结果, 后续的值可能会有负值, 不是softmax的概率值
  		output = self.softmax(self.out(output[0]))
  		print('output--->', output.shape, output)
  		return output, hidden
  	
  	
  if __name__ == '__main__':
  	# 获取数据
  	(english_word2index, english_index2word, english_word_n,
  	 french_word2index, french_index2word, french_word_n, my_pairs) = my_getdata()
  	# 创建张量数据集
  	my_dataset = MyPairsDataset(my_pairs, english_word2index, french_word2index)
  	# 创建数据加载器
  	# batch_size: 当前设置为1, 因为句子长度不一致
  	my_dataloader = DataLoader(dataset=my_dataset, batch_size=1, shuffle=True)
  	# 创建编码器对象
  	my_encoderrnn = EncoderRNN(input_size=english_word_n, hidden_size=256).to(device=device)
  	# 创建解码器对象
  	output_size = french_word_n
  	hidden_size = 256
  	my_decoderrnn = DecoderRNN(output_size, hidden_size).to(device)
  	for i, (x, y) in enumerate(my_dataloader):
  		# 编码器进行编码 一次性喂数据
  		# 初始化隐藏状态值
  		hidden = my_encoderrnn.inithidden()
  		encoder_output, hn = my_encoderrnn(x, hidden)
  		print('encoder_output--->', encoder_output.shape)
  		print('hn--->', hn.shape, hn)
  		
  		# 解码器进行解码, 自回归, 一个一个token进行解码
  		for j in range(y.shape[1]) :
  			# 获取当前预测token时间步的输入x(等同于上一时间步的预测y)
  			# 当前以真实y中的每个token作为输入, 模拟解码器的界面过程, 实际上第一个输入token一定是起始符号
  			tmp_y = y[0][j].view(1, -1)
  			# 进行解码
  			# 初始的隐藏状态值=编码器最后一个时间步的隐藏状态值
  			my_decoderrnn(tmp_y, hn)
  			break
  		break
  ```

- 构建基于GRU和Attention的解码器

  ```python
  # 带加性注意力机制的解码器
  class AttnDecoderRNN(nn.Module):
  	# todo:1- 定义构造方法 init
  	def __init__(self, output_size, hidden_size, dropout_p=0.2, max_length=MAX_LENGTH):
  		super().__init__()
  		# 初始化词嵌入层的输入维度和全连接层的输出维度一致
  		self.output_size = output_size
  		# 初始化编码器解码器隐藏层维度属性  解码器的第一个隐藏状态值=编码器的最后一个隐藏状态值
  		# 初始化词嵌入层维度属性  共享
  		self.hidden_size = hidden_size
  		# 初始化最大句子长度属性 -> 所有句子 c的长度固定
  		self.max_length = max_length
  		# 初始化dropout概率属性
  		self.dropout_p = dropout_p
  		# 初始化 embeding层
  		self.embedding = nn.Embedding(num_embeddings=self.output_size, embedding_dim=self.hidden_size)
  		# 初始化 gru层
  		self.gru = nn.GRU(input_size=self.hidden_size, hidden_size=self.hidden_size, batch_first=True)
  		# 初始化 全连接层
  		self.out = nn.Linear(in_features=self.hidden_size, out_features=self.output_size)
  		self.softmax = nn.LogSoftmax(dim=-1)
  		
  		# 初始化注意力机制中两个线性层
  		"""
  		q:解码器当前预测时间步的隐藏状态值
  		k:解码器当前预测时间步的上一时间步隐藏状态值
  		v:编码器的output输出
  		q,k,v三个特征维度相同 都是hidden_size
  		"""
  		# in_features: q和k的特征维度拼接
  		# out_features: 后续权重概率矩阵->(1, 1, max_len) 和 V矩阵相乘 V->(1, max_len, hidden_size)
  		self.attn = nn.Linear(in_features=self.hidden_size + self.hidden_size, out_features=self.max_length)
  		# in_features: q和c的特征维度拼接
  		# out_features: 输出的维度和gru层的输入维度保持一致
  		self.attn_combine = nn.Linear(in_features=self.hidden_size + self.hidden_size, out_features=self.hidden_size)
  		# 初始化dropout层
  		self.dropout = nn.Dropout(p=self.dropout_p)
  	
  	# todo:2- 定义前向传播方法 forward
  	def forward(self, input, hidden, encoder_outputs):
  		"""
  		前向传播计算
  		:param input: q, 解码器当前预测时间步的输入x, 也是上一个时间步预测的输出y
  		:param hidden: k, 上一个时间步的隐藏状态值, 第一个时间步的上一个隐藏状态值=编码器最后一个时间步的隐藏状态值
  		:param encoder_outputs: v, 编码器的输出 output, 后续是统一长度都为10, 10个token, 不足10个token用0填充
  		:return: 预测词表概率向量, 当前时间步的隐藏状态值, 权重概率矩阵
  		"""
  		# 2-1 词嵌入操作
  		embedded = self.embedding(input)
  		# 使用dropout防止过拟合
  		embedded = self.dropout(embedded)
  		print('embedded--->', embedded.shape, embedded)
  		
  		# 2-2 计算权重分数矩阵, 之后再计算权重概率矩阵
  		# q和k在特征维度轴拼接 + 线性计算 + softmax计算
  		# embedded[0]: 获取二维向量表示, 线性层一般接收二维数据
  		attn_weights = torch.softmax(self.attn(torch.cat(tensors=[embedded[0], hidden[0]], dim=1)), dim=-1)
  		print('attn_weights--->', attn_weights.shape, attn_weights)
  		# print(torch.sum(input=attn_weights))
  		
  		# 2-3 计算动态c, 加权求和  权重概率矩阵和v进行三维矩阵乘法
  		# bmm() 三维矩阵乘法, 目前attn_weights和encoder_outputs二维矩阵
  		attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
  		print('attn_applied--->', attn_applied.shape, attn_applied)
  		
  		# 2-4 q和动态c融合线性计算, 得到gru的输入x
  		# unsqueeze():得到三维数据, gru的输入x的形状要求
  		output = self.attn_combine(torch.cat(tensors=[embedded[0], attn_applied[0]], dim=1)).unsqueeze(0)
  		print('output--->', output.shape, output)
  		# relu激活函数, 非线性因素
  		output = torch.relu(output)
  		
  		# 2-5 gru层操作
  		output, hidden = self.gru(output, hidden)
  		print('output--->', output.shape, output)
  		print('hidden--->', hidden.shape, hidden)
  		
  		# 2-6 全连接层操作
  		output = self.softmax(self.out(output[0]))
  		print('output--->', output.shape, output)
  		return output, hidden, attn_weights
  
  
  if __name__ == '__main__':
  	# 获取数据
  	(english_word2index, english_index2word, english_word_n,
  	 french_word2index, french_index2word, french_word_n, my_pairs) = my_getdata()
  	# 创建张量数据集
  	my_dataset = MyPairsDataset(my_pairs, english_word2index, french_word2index)
  	# 创建数据加载器
  	# batch_size: 当前设置为1, 因为句子长度不一致
  	my_dataloader = DataLoader(dataset=my_dataset, batch_size=1, shuffle=True)
  	# 创建编码器对象
  	my_encoderrnn = EncoderRNN(input_size=english_word_n, hidden_size=256).to(device=device)
  	# 创建解码器对象
  	output_size = french_word_n
  	hidden_size = 256
  	# my_decoderrnn = DecoderRNN(output_size, hidden_size).to(device)
  	
  	# 创建带attn的解码器对象
  	my_attndecoderrnn = AttnDecoderRNN(output_size, hidden_size).to(device)
  	for i, (x, y) in enumerate(my_dataloader):
  		# print('x--->', x.shape)
  		# 编码器进行编码 一次性喂数据
  		# 初始化隐藏状态值
  		hidden = my_encoderrnn.inithidden()
  		encoder_output, hn = my_encoderrnn(x, hidden)
  		print('encoder_output--->', encoder_output.shape, encoder_output)
  		# print('hn--->', hn.shape, hn)
  		
  		# 获取填充成最大程度的编码器c或者output
  		# 初始化全0的张量 形状(10, 256) [[0,0,0,0,0,0,...],[],[]]
  		encoder_output_c = torch.zeros(size=(MAX_LENGTH, my_encoderrnn.hidden_size), device=device)
  		# 将encoder_output真实值赋值到encoder_output_c对应位置
  		for idx in range(x.shape[1]):
  			encoder_output_c[idx] = encoder_output[0][idx]
  		print('encoder_output_c--->', encoder_output_c.shape, encoder_output_c)
  		# 解码器进行解码, 自回归, 一个一个token进行解码
  		for j in range(y.shape[1]):
  			# 获取当前预测token时间步的输入x(等同于上一时间步的预测y)
  			# 当前以真实y中的每个token作为输入, 模拟解码器的界面过程, 实际上第一个输入token一定是起始符号
  			tmp_y = y[0][j].view(1, -1)
  			# 进行解码
  			# 初始的隐藏状态值=编码器最后一个时间步的隐藏状态值
  			# my_decoderrnn(tmp_y, hn)
  			# hn:编码器端最后一个时间步的隐藏状态值, 也是解码器端第一个时间步的初始的隐藏状态值
  			print('hn--->', hn.shape, hn)
  			output, hidden, attn_weights = my_attndecoderrnn(tmp_y, hn, encoder_output_c)
  			print('=' * 80)
  			print('output--->', output.shape, output)
  			print('hidden--->', hidden.shape, hidden)
  			print('attn_weights--->', attn_weights.shape, attn_weights)
  			break
  		break
  ```
