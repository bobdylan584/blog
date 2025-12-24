---
date: 2025-10-14
title: Agent的工具封装和调用
categories: [Agent, Agent的工具封装和调用]
tag: Agent
---

# Agent的工具封装和调用

## FunctionCall

server端，封装tool工具

client端，调用工具

### 工具的三种定义方式：

1、定义 JSON 格式的工具 schema

2、@tool装饰器定义

3、定义工具类

### 调用工具的两种方式：

1、把工具绑到大模型上

2、把工具和大模型以及提示词一起绑到智能体上



pydantic工具定义（把工具绑到大模型上，让大模型自动推理应该用哪一个工具，并设计逻辑返回，工具处理好的，放到json中对应的答案字段）

````python
# todo: 第一步： 定义工具类：
class Add(BaseModel):
class Multiply(BaseModel):
# 定义 类名
tools = [Add,Multiply]

# todo: 第二步：初始化模型
llm = ChatOpenAI(base_url=conf.base_url,
                 api_key=conf.api_key,
                 model=conf.model_name,
                 temperature=0.1)
# 绑定工具，允许模型自动选择工具
llm_with_tools = llm.bind_tools(tools, tool_choice="auto")

# todo: 第三步：调用回复
query = "2+1等于多少？"
messages = [HumanMessage(query)]

try:
    # todo: 第一次调用
    ai_msg = llm_with_tools.invoke(messages)
    messages.append(ai_msg)
    print(f"\n第一轮调用后结果：\n{messages}")
    print(f"\nai_msg.tool_calls打印的结果：\n{ai_msg.tool_calls}")


    # 处理工具调用
    # 判断消息中是否有tool_calls，以判断工具是否被调用:(注释：has attribution有没有该属性,并且ai_msg.tool_calls不为空)
    # 如果ai_msg中有属性tool_calls，并且不为空则：
    if hasattr(ai_msg, 'tool_calls') and ai_msg.tool_calls:
        for tool_call in ai_msg.tool_calls:# 遍历里面的参数，列表中只有一个（字典），也就是遍历一个：[{'name': 'add', 'args': {'a': 2, 'b': 1}, 'id': 'call_38c13ee30389422bbde17f', 'type': 'tool_call'}],
            # todo: 处理工具类调用，
            #  基于工具名称获取对应的类
            func_class = {"add": Add, "multiply": Multiply}[tool_call["name"].lower()]
            # 实例化工具类，并调用invoke方法
            func_obj = func_class(**tool_call["args"])
            tool_result = func_obj.invoke(tool_call["args"])
            # 因为原始的函数使用了注解，现在就是一个被langchain封装后的方法，调用时需要使用invoke进行调用

            # 则tool_output = 3
            messages.append(ToolMessage(content=tool_result, tool_call_id=tool_call["id"]))
        print(f"\n第二轮  message中增加tool_output 之后：\n{messages}")

        # todo: 第二次调用，将工具结果传回模型以生成最终回答
        final_response = llm_with_tools.invoke(messages)
        print(f"\nfinal_response打印结果：\n{final_response}")
        print(f"\n最终模型响应：\n{final_response.content}")
    else:
        print("模型未生成工具调用，直接返回文本:")
        print(ai_msg.content)
except Exception as e:
    print(f"模型调用失败: {str(e)}")
````



3、智能体调用工具：（使用langchain里面的tool工具作为注解，agent方法，初始化和调用智能体）

````python
from langchain_core.tools import tool
from langchain.agents import  initialize_agent,AgentType
# todo: 第一步：定义工具函数
@tool
def add()
@tool
def multiply()
tools = [add,multiply]
# todo: 第二步：初始化模型
llm = ChatOpenAI(base_url=conf.base_url,
                 api_key=conf.api_key,
                 model=conf.model_name,
                 temperature=0.1)
# todo: 第三步：创建智能体
agent = initialize_agent(tools,llm,AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
# todo: 第四步：调用回复
result = agent.invoke("2+1等于多少？")
print(f'result-->{result}')
````



## MCP

### MCP工具的使用：

1、定义服务端封装工具逻辑

2、定义客户端调用逻辑

### 三种不同传输方式：

1、stdio（标准输入输出，把服务器端当做客户端的子进程，在客户端一起吊起，在本地调试时用）

2、sse（基于http协议，server-sent-event，服务器端向客户端发送信息）

3、streamable http（流失输出，客户端和服务端双向发送信息）

### 客户端的两种不同调用方式

1、直接调用

2、agent调用





## Python A2A

python a2a支持mcp和A2A协议

sse、streamable、stdio三种服务器中使用的三方包都是：from mcp.server.fastmcp import FastMCP。但是也可以用

```
from python_a2a.mcp import FastMCP
from python_a2a.mcp import create_fastapi_app
```

区别：

1、多了create_fastapi_app，需要创建mcp实例

2、MCP的服务端和客户端的实例化，都不需要写url和地址了，默认就有