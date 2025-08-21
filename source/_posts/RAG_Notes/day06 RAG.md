---
title: day06 RAG
date: 2023-09-26 21:21:51
tag: RAG_Notes
---



# day06 RAG

Milvus

配置文件，

文档处理模块

日志书写

原始子块，txt

sparse vector

parent content

source

timestamp

检索精准。

分块检索，

父块

子块

子块的文档切分器

父类的文档切分器

基于父块，切

文档的处理模块

创建并加载集合

转换成向量并存储

trainer.train()

model

 args

帮助简化训练流程。

处理数据（※）

帮助跟随节奏：

敲注释，敲print信息。

```
检查模型路径是否存在
加载预训练模型
将模型移到指定设备
记录加载成功的日志

初始化新模型
num labels=4；就是多分类
default：就是2
将模型移到指定设备
记录初始化模型的日志

保存模型（save_model)


创建dataset对象

训练bert分类模型
# 加载数据集
with open万能读取方式
json.loads(value) for value in f.readlines()

print:
texts[:2]
labels[:2]
len(train_texts)
len(val_texts)
#预处理
##数据预处理
truncation=True,
padding= 'max_length'
max_length = 128
return_tensors='pt'
print(encoding)
print(encoding['input_ids'].shape)
## def create_dataset(self,encodings,labels)
	class dataset(torch.utils.data.dataset)
    	def __init__(self,encoding,labels)
    		super().__init__()
    		
len(train_dataset)
train_dataset[1]

main
query classifer = QueryClassifier()
data_file
query_classify.train_model()


```



## 上午回顾

realize rag_prompt method

provide a context

 let model answer the question based on context

normally, input with json format

defined sayed question 

retrievel context with query

target :

coplicate into simple

simplize

there is prompt template class in langchain pack

write queryclassifier

normal common

or professional query

made up by LLM

giving a prompt mudule

splite 5000 into serveral batch

warning:farmat warong

at leaset

simplize script 

load_model加载模型

保存模型

create dataset类

实例化对象

```
数据划分
预处理
创建数据集
设置训练参数
取出train dataset的一个样本呢数据
train dataset字典格式
**等价于变量赋值。

train_args=TrainingArguments(
输出路径
训练轮数=3
GPU的批次大小

学习率预热（线性学习率预热，从0预热到指定的步长的这个值）
预热就是为学习率准备的
权重衰减系数（adam），会有默认值
log文件夹
每个variable值打印一次
“epoch”，每一轮都去验证一下模型的指标
“epoch”每一轮都去保存一下；梯度检查点（会有epoch=3个模型）
总结每轮都会检查一次，并保存一次

最后加载最优的模型
只保存一个检查点（save total limit=1）就是对每轮的模型参数进行覆写。最后只会留下最后一个模型，约束检查点的个数，最终只保留一个。
metric_for best-model=“eval-loss“看最终的损失值
fp16=false 禁用混合精度
)
# 初始化 trainner
有一个参数computer metrics = self。compute-metrics
compute_metrics(self,eval_pred):
	logits,labels = eval_pred
	logits:[8,2]模型预测的结果
	labels:[8]
	
# 初始化trainer（书写代码）
	
"""
如果是自己定义的模型
torch.save(model.state_dict(),'path')
model.load_state_dict(torch.load('pth'))
***看视频面试，操作一次，copy一次，看视频一次。***

"""
打印结果分析：
loss：查看loss是否能够正常下降
grad-norm：梯度裁剪（防止梯度爆炸）
learning-rate：学习率
epoch
338；338/8 *轮数3   =   147=迭代的部署
每隔10步打印一次日志；


evaluate-model（对训练好的模型进行评估）
encoding
输入的数据必须先分词
pytorch 张量 底层是数组
predict——category
检查模型是否加载
encoding是字典，
对查询进行编码

encoding = {k：v。to（self。device） for k，v in }
logits
```

## 知识串讲

```
流程图：意图识别；通用知识则大模型直接回答。是专业知识，则调用rag qa系统。
将上下文和模版做一个凭借。
查询分类
意图识别的模型（2分类）
写一个提示词（规则）让大模型生成样本。
套用预训练模型
trainer方法实现训练脚本
将模型重新加载然后预测
```

