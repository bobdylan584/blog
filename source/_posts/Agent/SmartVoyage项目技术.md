---
date: 2025-07-15
title: SmartVoyage项目技术
categories: [Agent, SmartVoyage项目技术]
tag: Agent
---

# P01_项目技术

# 一、Function Call 函数调用

------

## 1 什么是Function Call【理解】

概念：大模型基于具体任务，智能决策何时需要调用某个函数，同时返回符合函数参数的 JSON对象。

能力获得的方式：基于训练来得到的，所以并不是所有大模型都具有Function Call能力。

优势：信息实时性、数据局限性、功能扩展性。



## 2 Function Call 工作原理【理解】

主要步骤：

1. 用户(client)发请求提示词，chat server将提示词和可以调用的函数发送给大模型
2. GPT模型根据用户的提示词，判断是用普通文本还是函数调用的格式回复我们的服务(chat server)
3. 如果是函数调用格式，那么chat server就会执行这个函数，并且将结果返回给GPT
4. 然后模型使用提供的数据，用连贯的文本响应。



## 3 Function Call 使用方式

### 3.1 自定义tool结构【熟悉】

代码：

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage

from agent_learn.config import Config

conf = Config()
# print(f'conf.model_name: {conf.model_name}')

# todo: 第一步：定义工具函数
def add(a: int, b: int) -> int:
    """
    将数字a与数字b相加
    Args:
        a: 第一个数字
        b: 第二个数字
    """
    return a + b


def multiply(a: int, b: int) -> int:
    """
    将数字a与数字b相乘
    Args:
        a: 第一个数字
        b: 第二个数字
    """
    return a * b

# 定义 JSON 格式的工具 schema
tools = [
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "将数字a与数字b相加",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {
                        "type": "integer",
                        "description": "第一个数字"
                    },
                    "b": {
                        "type": "integer",
                        "description": "第二个数字"
                    }
                },
                "required": ["a", "b"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "将数字a与数字b相乘",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {
                        "type": "integer",
                        "description": "第一个数字"
                    },
                    "b": {
                        "type": "integer",
                        "description": "第二个数字"
                    }
                },
                "required": ["a", "b"]
            }
        }
    }
]

# todo: 第二步：初始化模型
llm = ChatOpenAI(base_url=conf.base_url,
                 model=conf.model_name,
                 api_key=conf.api_key,
                 temperature=0.2)
# 绑定工具，允许模型自动选择工具
llm_with_tools = llm.bind_tools(tools, tool_choice="auto")

# todo: 第三步：调用回复
query = "2+1等于多少？"
# query = '什么是机器学习？'
# 使用列表的方式来存储对话信息
messages = [HumanMessage(query)]


try:
    # todo: 第一次调用
    ai_msg = llm_with_tools.invoke(messages)
    messages.append(ai_msg)
    print(f"\n第一轮调用后结果：\n{messages}")

    # 处理工具调用
    # 判断消息中是否有tool_calls，以判断工具是否被调用
    if hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls:
        for tool_call in ai_msg.tool_calls:
            # 基于工具名称获取对应的函数
            func = {"add": add, "multiply": multiply}[tool_call["name"].lower()]
            # 解析参数，并调用对应的函数
            tool_result = func(**tool_call["args"])
            # 将工具调用结果添加到messages中，用于下一次调用
            messages.append(ToolMessage(content=tool_result, tool_call_id=tool_call["id"]))
        print(f"\n工具调用结果添加到messages后：\n{messages}")

        # todo: 第二次调用，将工具结果传回模型以生成最终回答
        final_response = llm_with_tools.invoke(messages)
        print(f"\n最终模型响应：\n{final_response.content}")
    else:
        print("模型未生成工具调用，直接返回文本:")
        print(ai_msg.content)
except Exception as e:
    print(f"调用失败：{e}")
```



注意：

```properties
llm.invoke(messages, tools=tools, ...):
绑定方式: 直接在 .invoke() 调用中传入 tools 参数。这是一种临时、一次性的绑定方式，仅对本次调用有效。
调用方式: 如果你想再次调用模型并使用工具，你必须在下一次 .invoke() 调用中再次传递 tools 参数。
适用场景: 适用于简单、单次的工具调用需求，
```



### 3.2 装饰器tool方式【掌握】

以下是代码通过装饰器@tool的方式进行工具定义：

**定义方式**：通过 `@tool` 装饰器直接装饰一个普通的 Python 函数，比如 `add` 和 `multiply`。

**工作原理**：`@tool` 装饰器会自动根据函数签名（如 `a: int, b: int`）和文档字符串生成一个完整的工具定义（schema），包括工具名称、描述和参数结构。

**优势**：

- **简洁高效**：这是最简单、最 Pythonic 的方式，几乎不需要额外的样板代码。你只需编写核心函数逻辑，工具定义部分由框架自动处理。
- **自动化**：LangChain 的工具系统会自动处理工具的封装和调用，包括基本的参数类型验证。

------

**使用的方法：**

1）添加注解

from langchain_core.tools import tool

在函数名上使用@tool进行注解

2）定义tools

tools = [add, multiply]

3）解析参数，调用函数

```python
 # 解析参数，并调用对应的函数【因为原始的函数使用了注解，现在就是一个被langchain封装后的方法，调用时需要使用invoke进行调用】
 tool_result = func.invoke(tool_call["args"])
```



代码如下：

```python
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage

from agent_learn.config import Config

conf = Config()
# print(f'conf.model_name: {conf.model_name}')

# todo: 第一步：定义工具函数
@tool
def add(a: int, b: int) -> int:
    """
    将数字a与数字b相加
    Args:
        a: 第一个数字
        b: 第二个数字
    """
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """
    将数字a与数字b相乘
    Args:
        a: 第一个数字
        b: 第二个数字
    """
    return a * b

# 定义 JSON 格式的工具 schema
tools = [add, multiply]

# todo: 第二步：初始化模型
llm = ChatOpenAI(base_url=conf.base_url,
                 model=conf.model_name,
                 api_key=conf.api_key,
                 temperature=0.2)
# 绑定工具，允许模型自动选择工具
llm_with_tools = llm.bind_tools(tools, tool_choice="auto")

# todo: 第三步：调用回复
query = "2+1等于多少？"
# query = '什么是机器学习？'
# 使用列表的方式来存储对话信息
messages = [HumanMessage(query)]


try:
    # todo: 第一次调用
    ai_msg = llm_with_tools.invoke(messages)
    messages.append(ai_msg)
    print(f"\n第一轮调用后结果：\n{messages}")

    # todo: 处理工具调用
    # 判断消息中是否有tool_calls，以判断工具是否被调用
    if hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls:
        for tool_call in ai_msg.tool_calls:
            # 基于工具名称获取对应的函数
            func = {"add": add, "multiply": multiply}[tool_call["name"].lower()]
            # 解析参数，并调用对应的函数【因为原始的函数使用了注解，现在就是一个被langchain封装后的方法，调用时需要使用invoke进行调用】
            tool_result = func.invoke(tool_call["args"])
            # 将工具调用结果添加到messages中，用于下一次调用
            messages.append(ToolMessage(content=tool_result, tool_call_id=tool_call["id"]))
        print(f"\n工具调用结果添加到messages后：\n{messages}")

        # todo: 第二次调用，将工具结果传回模型以生成最终回答
        final_response = llm_with_tools.invoke(messages)
        print(f"\n最终模型响应：\n{final_response.content}")
    else:
        print("模型未生成工具调用，直接返回文本:")
        print(ai_msg.content)
except Exception as e:
    print(f"调用失败：{e}")
```



### 3.3 pydantic的tool方式【了解】

通过严格数据校验pydantic进行工具定义：

**定义方式**：创建一个继承自 `BaseModel` 的类，用类型注解和 `Field` 定义工具的参数。同时，需要在类中手动实现一个 `invoke` 方法来包含工具的执行逻辑。

 **工作原理**：

- **数据验证**：Pydantic 提供了强大的数据验证功能。当工具被调用时，它会自动验证传入的参数是否符合你在 `BaseModel` 中定义的类型和约束。
- **手动实现**：与 `@tool` 不同，Pydantic 本身不提供工具的执行逻辑。因此，你必须显式地编写 `invoke` 方法来处理参数并返回结果。

 **优势**：

- **强大的数据验证**：Pydantic 提供了比 `@tool` 更细粒度和更丰富的参数验证功能，可以定义更复杂的约束。
- **高度可控**：由于 `invoke` 方法是手动实现的，你可以完全控制工具的执行逻辑，例如添加复杂的预处理、错误处理或自定义逻辑。
- **清晰的结构**：工具的参数定义和执行逻辑被封装在一个类中，使得代码结构更加清晰。

------

代码如下：

```python
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage
from pydantic.v1 import BaseModel, Field

from agent_learn.config import Config

conf = Config()
# print(f'conf.model_name: {conf.model_name}')

# todo: 第一步：定义工具函数
class Add(BaseModel):
    """
    将两个数字相加
    """
    a: int = Field(..., description="第一个数字")
    b: int = Field(..., description="第二个数字")

    def invoke(self, args):
        # 验证参数
        tool_instance = self.__class__(**args)  # 自动验证 a 和 b
        return tool_instance.a + tool_instance.b

class Multiply(BaseModel):
    """
    将两个数字相乘
    """
    a: int = Field(..., description="第一个数字")
    b: int = Field(..., description="第二个数字")

    def invoke(self, args):
        # 验证参数
        tool_instance = self.__class__(**args)  # 自动验证 a 和 b
        return tool_instance.a * tool_instance.b

# 定义 JSON 格式的工具 schema
tools = [Add, Multiply]

# todo: 第二步：初始化模型
llm = ChatOpenAI(base_url=conf.base_url,
                 model=conf.model_name,
                 api_key=conf.api_key,
                 temperature=0.2)
# 绑定工具，允许模型自动选择工具
llm_with_tools = llm.bind_tools(tools, tool_choice="auto")

# todo: 第三步：调用回复
query = "2.1 + 1 等于多少？"
# query = '什么是机器学习？'
# 使用列表的方式来存储对话信息
messages = [HumanMessage(query)]


try:
    # todo: 第一次调用
    ai_msg = llm_with_tools.invoke(messages)
    messages.append(ai_msg)
    print(f"\n第一轮调用后结果：\n{messages}")

    # todo: 处理工具调用
    # 判断消息中是否有tool_calls，以判断工具是否被调用
    if hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls:
        for tool_call in ai_msg.tool_calls:
            # 基于工具名称获取对应的类
            func_class = {"add": Add, "multiply": Multiply}[tool_call["name"].lower()]
            # 实例化工具类，并调用invoke方法
            func_obj = func_class(**tool_call["args"])
            tool_result = func_obj.invoke(tool_call["args"])
            # 将工具调用结果添加到messages中，用于下一次调用
            messages.append(ToolMessage(content=tool_result, tool_call_id=tool_call["id"]))
        print(f"\n工具调用结果添加到messages后：\n{messages}")

        # todo: 第二次调用，将工具结果传回模型以生成最终回答
        final_response = llm_with_tools.invoke(messages)
        print(f"\n最终模型响应：\n{final_response.content}")
    else:
        print("模型未生成工具调用，直接返回文本:")
        print(ai_msg.content)
except Exception as e:
    print(f"调用失败：{e}")
```

------

**总结：**

| 特性       | JSON Schema                                  | @tool 装饰器                   | Pydantic                                     |
| ---------- | -------------------------------------------- | ------------------------------ | -------------------------------------------- |
| 定义方式   | 手动编写 Python 字典（JSON Schema）          | 装饰 Python 函数               | 继承 Pydantic BaseModel                      |
| 自动化程度 | 低：完全手动定义和分发                       | 高：自动生成 Schema 和调用逻辑 | 中等：自动验证数据，但需手动实现 invoke      |
| 数据验证   | 需要手动验证或依赖外部库                     | 基础类型检查                   | 强大：提供丰富的验证功能                     |
| 适用场景   | 需要与其他系统集成、通用性和最大灵活性的场景 | 快速开发、简单工具、原型验证   | 需要复杂数据验证、清晰结构和自定义逻辑的场景 |



## 4 Agent 调用 tool【熟悉】

Agent（智能体）是一种能够感知环境、进行决策和执行动作的智能实体。从大模型的角度来看，**Agent其实就是基于大模型的语义理解和推理能力，让大模型拥有解决复杂问题时的任务规划能力，并调用外部工具来执行各种任务，并且能够保留“记忆”的一个智能体**。

> Agent = 大模型 + 任务规划（Planning） + 使用外部工具执行任务（Tools&Action） + 记忆（Memory）

Agent的核心就是大模型，它调用工具的方式通常通过Function Call实现，不够很多的Agent框架对内部的调用过程进行了封装，所以更易使用。

------

代码如下：

```python
from langchain.agents import initialize_agent, AgentType
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage

from agent_learn.config import Config

conf = Config()
# print(f'conf.model_name: {conf.model_name}')

# todo: 第一步：定义工具函数
@tool
def add(a: int, b: int) -> int:
    """
    将数字a与数字b相加
    Args:
        a: 第一个数字
        b: 第二个数字
    """
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """
    将数字a与数字b相乘
    Args:
        a: 第一个数字
        b: 第二个数字
    """
    return a * b

# 定义 JSON 格式的工具 schema
tools = [add, multiply]

# todo: 第二步：初始化模型
llm = ChatOpenAI(base_url=conf.base_url,
                 model=conf.model_name,
                 api_key=conf.api_key,
                 temperature=0.2)

# todo: 第三步：创建Agent
agent = initialize_agent(tools, llm, AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# todo: 第四步：调用回复
query = "2+1等于多少？"
result = agent.invoke(query)
print(f'result-->{result}')
```



# 二、MCP协议

## 1 背景 【了解】

使用function call时，需要对工具的描述，这是一个繁琐而复杂的过程。MCP的出现就是将这些工具的描述和调用进行统一。



## 2 什么是MCP协议【理解】

MCP（Model Context Protocol，模型上下文协议）是由 Anthropic 在2024年1月提出的一套开放协议，旨在实现大型语言模型（LLM）与外部数据源和工具的无缝集成，用来在大模型和数据源之间建立安全双向的链接。



### 2.1 MCP 核心架构

MCP协议有两个核心角色：客户端与服务端。

**MCP服务端 (Tool Provider)：**

- **角色**：工具的提供者。
- **职责**：将一个或多个本地函数（例如，Python函数）包装起来，通过一个标准的MCP接口暴露出去。它监听来自客户端的请求，执行对应的函数，并返回结果。
- **例子**：一个天气查询服务、一个数学计算服务、一个数据库访问服务。

**MCP客户端 (Tool Consumer)**：

- **角色**：工具的调用者或消费者。
- **职责**：连接到MCP服务端，查询可用的工具列表（自发现），并根据需要调用这些工具。
- **例子**：大模型Agent、自动化脚本、任何需要远程执行功能的应用程序。

<img src="https://bob-blog-image.oss-cn-shanghai.aliyuncs.com/Agent/MCP3.jpg" alt="MCP3" style="zoom:67%;" />

> MCP 主机（MCP Hosts）指的是发起请求的 LLM 应用程序。MCP 客户端（MCP Clients）指的是在主机程序内部的一个对象。

### 2.2 MCP 工具调用流程

**步骤 1：客户端注册并连接 MCP Server**

- MCP Client 启动后，根据配置文件或命令参数连接多个 MCP Server。

- 每个 Server 都会返回一份工具描述列表（Tool Manifest），包括：

    ```json
    [
      {
        "name": "query_mysql",
        "description": "执行 SQL 查询",
        "input_schema": {...},
        "output_schema": {...}
      }
    ]
    ```

- Client 将这些工具的元信息缓存并上报给 LLM，使大模型“知道”有哪些可用工具。

**步骤 2：LLM 接收用户输入并决定调用工具**

- 用户输入请求（如：“帮我查一下 users 表中有多少行数据”）。

- LLM 分析语义后，判断需要使用 `query_mysql` 工具。

- LLM 生成 function calling 格式的调用指令：

    ```json
    {
      "name": "query_mysql",
      "arguments": {
        "sql": "SELECT COUNT(*) FROM users;"
      }
    }
    ```

**步骤 3：MCP Client 执行工具调用**

- MCP Client 收到该调用后，匹配到对应的 Server。

- 按协议通过 stdio 或 WebSocket 将请求发送给 MCP Server，例如：

    ```json
    {
      "type": "tool_call",
      "tool": "query_mysql",
      "args": {"sql": "SELECT COUNT(*) FROM users;"}
    }
    ```

**步骤 4：MCP Server 执行工具逻辑**

- MCP Server 内部执行工具逻辑（例如运行 SQL 查询）。

- 生成结果：

    ```json
    {
      "result": [{"count": 520}]
    }
    ```

- 将结果通过相同的通信通道返回给 MCP Client。

**步骤 5：结果回传给 LLM**

- MCP Client 接收结果，并包装为 ToolMessage 发送回 LLM。

- LLM 读取结果上下文，再生成最终自然语言回答：

    > “数据库中共有 520 条用户记录。”



### 2.3 MCP的通信传输方式

MCP协议本身与传输方式无关，MCP 主要有三种通信传输方式：stdio、基于HTTP的SSE和Streamable。

**（1）stdio (标准输入/输出)**

- 类型：stdio一种非常经典和简单的进程间通信（IPC）方式。客户端启动服务端作为一个子进程。

- 工作原理: 客户端通过写入子进程的 **标准输入 (stdin)**  来发送请求，并通过读取子进程的 **标准输出 (stdout)**  来获取响应。这种方式简单高效，无需网络开销。

- 适用场景：非常适合在**本地环境**中，将一个命令行工具或脚本快速封装成一个 MCP 服务。

**（2）SSE (Server-Sent Events)**

- 类型：SSE 是一种 **基于 HTTP 的单向推送协议** ，它允许服务器在保持连接开放的情况下，持续向客户端发送事件流。

- 工作原理：客户端发起一个 HTTP 请求，服务器接收请求并保持连接，然后以 text/event-stream 格式将响应数据流式传输给客户端。这在 MCP 中被用来实现请求与响应的通信。

- 适用场景： 适用于 **分布式或网络环境** ，当服务需要部署在远端，并通过网络供多个客户端访问时。

**（3）Streamable**

- 类型：Streamable-HTTP 是 MCP 提供的另一种基于 HTTP 的传输方式，它同样用于网络通信。

- 工作原理：客户端通过 HTTP 请求与服务器通信。与 SSE 的主要区别在于其传输格式和机制可能有所不同，比如Streamble的传输格式可以为任意格式，而SSE为特定格式；Streamble的通信方向可以为双向，而SSE只能是单向。

- 适用场景：与 SSE 类似，适用于需要**通过网络进行通信的分布式应用**。

**对比：**

| 传输方式         | stdio                          | SSE (Server-Sent Events)                   | Streamable-HTTP                                    |
| ---------------- | ------------------------------ | ------------------------------------------ | -------------------------------------------------- |
| 通信方向         | 双向（请求-响应）              | 单向（服务器推送到客户端）                 | 双向（双向流）                                     |
| 通信模式         | 本地进程间通信（IPC）          | 网络通信（长连接流）                       | 网络通信（双向流）                                 |
| 主要用途         | 封装本地命令行工具             | 仅需接收服务器更新的场景                   | 复杂的、需要实时双向流的场景                       |
| 是否容易丢失数据 | 由操作系统保证可靠性           | 由 TCP 协议保证可靠性                      | 由 TCP 协议保证可靠性                              |
| 关键优势         | 简单、高效、安全，无需网络开销 | 适用于简单、单向的数据推送，浏览器兼容性好 | 灵活性高，支持双向流式传输，提升大型任务的响应效率 |

------



## 3 mcp包使用

> 注意：以下代码需要安装langchain_mcp_adapters包，安装方式如下。
>
> pip install langchain-mcp-adapters --index-url https://pypi.org/simple

### 3.1 stdio传输方式【理解】

#### 服务端

```python
from mcp.server import FastMCP

# 获取mcp的对象
mcp = FastMCP('stoio_server', log_level='ERROR')


# 创建工具
@mcp.tool(
    name="query_high_frequency_question",
    description="从知识库中检索常见问题解答（FAQ）,返回包含问题和答案的结构化JSON数据。",
)
async def query_high_frequency_question() -> str:
    """
    高频问题查询
    """
    try:
        print("调用查询高频问题的tool成功！！")
        return "高频问题是: 恐龙是怎么灭绝的？"
    except Exception as e:
        print(f"Unexpected error in question retrieval: {str(e)}")
        raise


@mcp.tool(
    name="get_weather",
    description="查询天气"
)
async def get_weather() -> str:
    """
    查询天气的tools
    """
    try:
        print("调用查询天气的tools")
        return "北京的天气是多云"
    except Exception as e:
        print(f"Unexpected error in question retrieval: {str(e)}")
        raise


if __name__ == '__main__':
    mcp.run(transport='stdio')

```



#### 客户端（直接调用）

> 如果直接运行报错，可能是编码集的问题，可以尝试使用命令行的方式运行：
>
> set PYTHONIOENCODING=utf-8
>
> python client_direct.py 

```python
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import StdioServerParameters, ClientSession
from mcp.client.stdio import stdio_client
import asyncio

# 配置mcp服务器脚本路径
server_script = r".\stdio_server.py"

# 配置mcp服务器启动参数
server_parameters = StdioServerParameters(
    command = "python" if server_script.endswith(".py") else "node",
    args = [server_script],
)

# 定义mcp客户端
mcp_client = None

# 创建一个异步函数，来实现客户端的创建及使用
async def main():
    global mcp_client
    # 启动 MCP server，并通过标准输入输出建立异步连接。
    async with stdio_client(server_parameters) as (read, write):
        # 使用读写流创建MCP会话对象
        async with ClientSession(read, write) as session:
            # 初始化session
            await session.initialize()

            # 动态创建一个临时类 MCPClientHolder，把 session 放进去。这样就可以在函数外部通过 mcp_client.session 调用 MCP 工具
            mcp_client = type("MCPClientHolder", (), {"session": session})()

            # 从 session 自动获取 MCP server 提供的工具列表。
            tools = await load_mcp_tools(session)
            print(f"tools-->{tools}")

            # 通过session调用工具
            result = await session.call_tool("get_weather", {})
            print(f'result-->{result}')
    return # 终止进程


# 启动运行
if __name__ == '__main__':
    asyncio.run(main())

```



#### 客户端（agent调用）

```python
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI
from mcp import StdioServerParameters, ClientSession
from mcp.client.stdio import stdio_client
import asyncio

from agent_learn.config import Config

conf = Config()

# 创建模型
llm = ChatOpenAI(base_url=conf.base_url,
                 api_key=conf.api_key,
                 model=conf.model_name,
                 temperature=0.1)

# 配置mcp服务器脚本路径
server_script = r".\stdio_server.py"

# 配置mcp服务器启动参数
server_parameters = StdioServerParameters(
    command = "python" if server_script.endswith(".py") else "node",
    args = [server_script],
)

# 定义mcp客户端
mcp_client = None

# 创建一个异步函数，来实现客户端的创建及使用
async def main():
    global mcp_client
    # 启动 MCP server，并通过标准输入输出建立异步连接。
    async with stdio_client(server_parameters) as (read, write):
        # 使用读写流创建MCP会话对象
        async with ClientSession(read, write) as session:
            # 初始化session
            await session.initialize()

            # 动态创建一个临时类 MCPClientHolder，把 session 放进去。这样就可以在函数外部通过 mcp_client.session 调用 MCP 工具
            mcp_client = type("MCPClientHolder", (), {"session": session})()

            # 从 session 自动获取 MCP server 提供的工具列表。
            tools = await load_mcp_tools(session)
            print(f"tools-->{tools}")

            # 创建prompt模板
            # agent_scratchpad 这个参数是agent在进行推理的时候，自动生成的，用于记录agent的推理过程。
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", "你是一个乐于助人的助手，能够调用工具回答用户问题。"),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ])

            # 创建agent对象
            agent = create_tool_calling_agent(llm, tools, prompt_template)
            # 使用agent对象创建agent执行器
            agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

            # agent调用
            print("MCP客户端启动，输入'quit'退出")
            while True:
                # 接收用户查询
                query = input("\nQuery: ").strip()
                if query.lower() == "quit":
                    break
                try:
                    # 发送用户查询给代理，并打印  ainvoke 指的是异步调用
                    response = await agent_executor.ainvoke({"input": query})
                    print(f"response-->{response}")
                except Exception:
                    print("解析有问题")

    return # 终止进程

# 启动运行
if __name__ == '__main__':
    asyncio.run(main())
```





### 3.2 sse传输方式【理解】

#### 服务端

- 创建MCP对象不一样

    ```python
    # 获取mcp的对象
    mcp = FastMCP('stoio_server', log_level='ERROR', host="127.0.0.1", port=8001)
    ```

- 运行sse服务器

```python
 mcp.run(transport="sse")
```



```python
from mcp.server import FastMCP

# 获取mcp的对象
mcp = FastMCP('stoio_server', log_level='ERROR', host="127.0.0.1", port=8001)


# 创建工具
@mcp.tool(
    name="query_high_frequency_question",
    description="从知识库中检索常见问题解答（FAQ）,返回包含问题和答案的结构化JSON数据。",
)
async def query_high_frequency_question() -> str:
    """
    高频问题查询
    """
    try:
        print("调用查询高频问题的tool成功！！")
        return "高频问题是: 恐龙是怎么灭绝的？"
    except Exception as e:
        print(f"Unexpected error in question retrieval: {str(e)}")
        raise


@mcp.tool(
    name="get_weather",
    description="查询天气"
)
async def get_weather() -> str:
    """
    查询天气的tools
    """
    try:
        print("调用查询天气的tools")
        return "北京的天气是多云"
    except Exception as e:
        print(f"Unexpected error in question retrieval: {str(e)}")
        raise


if __name__ == '__main__':
    print("正在启动MCP SSE服务器...")
    print("SSE端点: http://localhost:8001/sse")
    print("按 Ctrl+C 停止服务器")

    try:
        # 运行SSE服务器
        mcp.run(transport="sse")
    except KeyboardInterrupt:
        print("\n服务器已停止")
    except Exception as e:
        print(f"服务器启动失败: {e}")

```



#### 客户端（直接调用）

- 直接配置url即可

```python
server_url = "http://localhost:8001/sse"
```

- 获取输入输出流，并创建客户端的会话对象

```python
    # 启动 MCP server，并通过标准输入输出建立异步连接。
    async with sse_client(url=server_url) as streams:
        # 使用读写流创建MCP会话对象
        async with ClientSession(*streams) as session:
```



```python
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import StdioServerParameters, ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
import asyncio

# MCP Server URL
server_url = "http://localhost:8001/sse"

# 定义mcp客户端
mcp_client = None

# 创建一个异步函数，来实现客户端的创建及使用
async def main():
    global mcp_client
    # 启动 MCP server，并通过标准输入输出建立异步连接。
    async with sse_client(url=server_url) as streams:
        # 使用读写流创建MCP会话对象
        async with ClientSession(*streams) as session:
            # 初始化session
            await session.initialize()

            # 动态创建一个临时类 MCPClientHolder，把 session 放进去。这样就可以在函数外部通过 mcp_client.session 调用 MCP 工具
            mcp_client = type("MCPClientHolder", (), {"session": session})()

            # 从 session 自动获取 MCP server 提供的工具列表。
            tools = await load_mcp_tools(session)
            print(f"tools-->{tools}")

            # 通过session调用工具
            result = await session.call_tool("get_weather", {})
            print(f'result-->{result}')

            # 使用client进行调用
            result2 = await mcp_client.session.call_tool("query_high_frequency_question", arguments={})
            print(f'result2-->{result2}')

    return # 终止进程

# 启动运行
if __name__ == '__main__':
    asyncio.run(main())

```



#### 客户端（agent调用）

```python
import os
import sys

from mcp.client.sse import sse_client

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI
from mcp import StdioServerParameters, ClientSession
from mcp.client.stdio import stdio_client
import asyncio

from agent_learn.config import Config

conf = Config()

# 创建模型
llm = ChatOpenAI(base_url=conf.base_url,
                 api_key=conf.api_key,
                 model=conf.model_name,
                 temperature=0.1)

# MCP Server URL
server_url = "http://localhost:8001/sse"

# 定义mcp客户端
mcp_client = None

# 创建一个异步函数，来实现客户端的创建及使用
async def main():
    global mcp_client
    # 启动 MCP server，并通过标准输入输出建立异步连接。
    async with sse_client(url=server_url) as streams:
        # 使用读写流创建MCP会话对象
        async with ClientSession(*streams) as session:
            # 初始化session
            await session.initialize()

            # 动态创建一个临时类 MCPClientHolder，把 session 放进去。这样就可以在函数外部通过 mcp_client.session 调用 MCP 工具
            mcp_client = type("MCPClientHolder", (), {"session": session})()

            # 从 session 自动获取 MCP server 提供的工具列表。
            tools = await load_mcp_tools(session)
            print(f"tools-->{tools}")

            # 创建prompt模板
            # agent_scratchpad 这个参数是agent在进行推理的时候，自动生成的，用于记录agent的推理过程。
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", "你是一个乐于助人的助手，能够调用工具回答用户问题。"),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ])

            # 创建agent对象
            agent = create_tool_calling_agent(llm, tools, prompt_template)
            # 使用agent对象创建agent执行器
            agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

            # agent调用
            print("MCP客户端启动，输入'quit'退出")
            while True:
                # 接收用户查询
                query = input("\nQuery: ").strip()
                if query.lower() == "quit":
                    break
                try:
                    # 发送用户查询给代理，并打印  ainvoke 指的是异步调用
                    response = await agent_executor.ainvoke({"input": query})
                    print(f"response-->{response}")
                except Exception:
                    print("解析有问题")

    return # 终止进程

# 启动运行
if __name__ == '__main__':
    asyncio.run(main())

```



### 3.3 streamable方式【熟悉】

#### 服务端

```python
from mcp.server.fastmcp import FastMCP

# 创建 MCP 实例，指定服务名称、日志级别、主机和端口
mcp = FastMCP("sdg", log_level="ERROR", host="127.0.0.1", port=8001)

@mcp.tool(
    name="query_high_frequency_question",
    description="从知识库中检索常见问题解答（FAQ）,返回包含问题和答案的结构化JSON数据。",
)
async def query_high_frequency_question() -> str:
    """
    高频问题查询
    """
    try:
        print("调用查询高频问题的tool成功！！")
        return "高频问题是: 恐龙是怎么灭绝的？"
    except Exception as e:
        print(f"Unexpected error in question retrieval: {str(e)}")
        raise

@mcp.tool(
    name="get_weather",
    description="查询天气"
)
async def get_weather() -> str:
    """
    查询天气的tools
    """
    try:
        print("调用查询天气的tools")
        return "北京的天气是多云"
    except Exception as e:
        print(f"Unexpected error in question retrieval: {str(e)}")
        raise


def main():
    """
    启动 Streamable HTTP 服务器。
    """
    print("正在启动MCP Streamable服务器...")
    print("服务器将在 http://localhost:8001 上运行")
    print("按 Ctrl+C 停止服务器")
    try:
        mcp.run(transport="streamable-http")  # 使用 streamable-http 传输方式
    except KeyboardInterrupt:
        print("\n服务器已停止")
    except Exception as e:
        print(f"服务器启动失败: {e}")


if __name__ == "__main__":
    main()
```



#### 客户端（直接调用）

```python
import asyncio
import logging
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

# 定义服务器地址
server_url = "http://127.0.0.1:8001/mcp"

# 定义mcp客户端
mcp_client = None

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,  # 提高日志级别以捕获更多信息
    format='[客户端] %(asctime)s - %(levelname)s - %(message)s'
)

async def main():
    global mcp_client
    logging.info(f"准备连接到 Streamable-HTTP 服务器: {server_url}")
    try:
        # 启动 MCP server，通过streamable建立连接
        async with streamablehttp_client(server_url) as (read, write, _):
            logging.info("连接已成功建立！")
            # 使用读写通道创建 MCP 会话
            async with ClientSession(read, write) as session:
                try:
                    await session.initialize()
                    logging.info("会话初始化成功，可以开始调用工具。")
                    # 动态创建一个临时类 MCPClientHolder，把 session 放进去。这样就可以在函数外部通过 mcp_client.session 调用 MCP 工具
                    mcp_client = type("MCPClientHolder", (), {"session": session})()

                    # 从 session 自动获取 MCP server 提供的工具列表。
                    tools = await load_mcp_tools(session)
                    # print(f"tools-->{tools}")

                    # 调用远程工具
                    logging.info("--> 正在调用工具: query_high_frequency_question")
                    response = await session.call_tool("query_high_frequency_question", {})
                    print(f"response-->{response}")
                    logging.info(f"<-- 收到响应: {response}")

                    print("-" * 30)

                    logging.info("--> 正在调用工具: get_weather")
                    response = await session.call_tool("get_weather", {})
                    print(f"response-->{response}")
                    logging.info(f"<-- 收到响应: {response}")
                except Exception as e:
                    logging.error(f"调用工具时发生错误: {e}", exc_info=True)
                    raise
    except Exception as e:
        logging.error(f"连接或会话初始化时发生错误: {e}", exc_info=True)
        logging.error("请确认服务端脚本已启动并运行在 http://127.0.0.1:8001/mcp")
        raise


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logging.error(f"客户端运行失败: {e}", exc_info=True)
```



#### 客户端（agent调用）

```python
import json
import logging
import asyncio
from langchain_openai import ChatOpenAI
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from agent_learn.config import Config

conf = Config()

# 创建模型
llm = ChatOpenAI(base_url=conf.base_url,
                 api_key=conf.api_key,
                 model=conf.model_name,
                 temperature=0.1)

# MCP 服务器的 Streamable-HTTP 连接地址
server_url = "http://127.0.0.1:8001/mcp"

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,  # 提高日志级别以捕获更多信息
    format='[客户端] %(asctime)s - %(levelname)s - %(message)s'
)

# 定义mcp客户端
mcp_client = None

async def run_agent():
    global mcp_client
    logging.info(f"准备连接到 Streamable-HTTP 服务器: {server_url}")
    # 启动 MCP server，通过streamable建立连接
    async with streamablehttp_client(server_url) as (read, write, _):
        logging.info("连接已成功建立！")
        # 使用读写通道创建 MCP 会话
        async with ClientSession(read, write) as session:
            try:
                await session.initialize()
                logging.info("会话初始化成功，可以开始加载工具。")
                # 动态创建一个临时类 MCPClientHolder，把 session 放进去。这样就可以在函数外部通过 mcp_client.session 调用 MCP 工具
                mcp_client = type("MCPClientHolder", (), {"session": session})()

                # 从 session 自动获取 MCP server 提供的工具列表。
                tools = await load_mcp_tools(session)
                # print(f"tools-->{tools}")

                # 创建 agent 的提示模板
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "你是一个乐于助人的助手，能够调用工具回答用户问题。"),
                    ("human", "{input}"),
                    ("placeholder", "{agent_scratchpad}"),
                ])

                # 构建工具调用代理
                agent = create_tool_calling_agent(llm, tools, prompt)

                # 创建代理执行器
                agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

                # 代理调用
                print("MCP客户端启动，输入'quit'退出")
                while True:
                    query = input("\nQuery: ").strip()
                    if query.lower() == "quit":
                        break
                    # 发送用户查询到 agent 并打印格式化响应
                    logging.info(f"处理用户查询: {query}")
                    try:
                        response = await agent_executor.ainvoke({"input": query})
                        print(f"response-->{response}")
                    except Exception:
                        print("解析有问题")
            except Exception as e:
                logging.error(f"会话初始化或工具调用时发生错误: {e}", exc_info=True)
                raise

if __name__ == "__main__":
    try:
        asyncio.run(run_agent())
    except Exception as e:
        logging.error(f"客户端运行失败: {e}", exc_info=True)
```



## 4 python_a2a包使用【熟悉】

#### 服务端

```python
import logging
import uvicorn
from python_a2a.mcp import FastMCP
from python_a2a.mcp import create_fastapi_app

# 配置日志，方便调试
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# 创建 MCP 服务器实例
mcp = FastMCP(
    name="MyMCPTools",
    description="提供高频问题和天气查询工具",
    version="1.0.0"
)


# 创建工具
@mcp.tool(
    name="query_high_frequency_question",
    description="从知识库中检索常见问题解答（FAQ）,返回包含问题和答案的结构化JSON数据。",
)
async def query_high_frequency_question(**kwargs) -> str:
    """
    高频问题查询
    """
    try:
        print(f"调用查询高频问题的tool成功！！传进来的参数为{kwargs}")
        # 返回结果可以是一个普通字符串，推荐是json字符串，将必要的信息封装到json字符串中。
        return '{"status": "success", "data": [{"question_id": 1, "question_text": "恐龙是怎么灭绝的？", "answer_text": "可能是小行星撞击", "category": "历史", "frequency_score": 0.9}]}'
    except Exception as e:
        print(f"Unexpected error in question retrieval: {str(e)}")
        raise


@mcp.tool(
    name="get_weather",
    description="查询天气"
)
async def get_weather(**kwargs) -> str:
    """
    查询天气的tools
    """
    try:
        print(f"调用查询天气的tools。传进来的参数为{kwargs}")
        return '{"status": "success", "data": "北京的天气是多云"}'
    except Exception as e:
        print(f"Unexpected error in question retrieval: {str(e)}")
        raise


# 启动服务器
def start_server():
    logger.info("=== MCP 服务器信息 ===")
    logger.info(f"名称: {mcp.name}")
    logger.info(f"描述: {mcp.description}")

    port = 8010
    app = create_fastapi_app(mcp)
    logger.info(f"启动 MCP 服务器于 http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == '__main__':
    start_server()
```



#### 客户端（直接调用）

```python
import asyncio
import logging

from python_a2a.mcp import MCPClient

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_mcp_tools():
    # 创建MCP客户端
    mcp_client = MCPClient(server_url="http://localhost:8010")
    try:
        # 获取可用的工具列表
        tools = await mcp_client.get_tools()
        print(f'MCP tools-->{tools}')

        # 直接调用工具
        result = await mcp_client.call_tool("query_high_frequency_question")
        print(f'MCP result-->{result}')

    except Exception as e:
        logger.error(f'执行报错，错误信息为：{e}')


if __name__ == '__main__':
    asyncio.run(test_mcp_tools())

```



#### 客户端（agent调用）

```python
from mcp.client.streamable_http import streamablehttp_client
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI
from mcp import ClientSession
import asyncio

from python_a2a import MCPClient, to_langchain_tool

from agent_learn.config import Config

conf = Config()

# 创建模型
llm = ChatOpenAI(base_url=conf.base_url,
                 api_key=conf.api_key,
                 model=conf.model_name,
                 temperature=0.1)


# 创建一个异步函数，来实现客户端的创建及使用
async def main():
    # 创建MCP客户端
    url = "http://localhost:8010"
    mcp_client = MCPClient(server_url=url)

    try:
        # 获取可用的工具列表
        tools = await mcp_client.get_tools()
        print(f'MCP tools-->{tools}')

        # 将 MCP tool 转成 LangChain 的工具
        get_weather_tool = to_langchain_tool(url, "get_weather")
        query_high_frequency_question = to_langchain_tool(url, "query_high_frequency_question")
        tools = [get_weather_tool, query_high_frequency_question]

        # 创建prompt模板
        # agent_scratchpad 这个参数是agent在进行推理的时候，自动生成的，用于记录agent的推理过程。
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "你是一个乐于助人的助手，能够调用工具回答用户问题。"),
            ("human", "{user_input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])

        # 创建agent对象
        agent = create_tool_calling_agent(llm, tools, prompt_template)
        # 使用agent对象创建agent执行器
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

        # agent调用
        print("MCP客户端启动，输入'quit'退出")
        while True:
            # 接收用户查询
            query = input("\nQuery: ").strip()
            if query.lower() == "quit":
                break
            try:
                # 发送用户查询给代理，并打印  ainvoke 指的是异步调用
                response = await agent_executor.ainvoke({"user_input": query})
                print(f"response-->{response}")
            except Exception:
                print("解析有问题")
    except Exception as e:
        print(f"Error: {e}")


# 启动运行
if __name__ == '__main__':
    asyncio.run(main())
```



# 三、Agent智能体

## 1 什么是 Agent 【理解】

Agent（智能体）是一种能够感知环境、进行决策和执行动作的智能实体。从大模型的角度来看，**Agent其实就是基于大模型的语义理解和推理能力，让大模型拥有解决复杂问题时的任务规划能力，并调用外部工具来执行各种任务，并且能够保留“记忆”的一个智能体**。

> Agent = 大模型 + 任务规划（Planning） + 使用外部工具执行任务（Tools&Action） + 记忆（Memory）



## 2 什么是Agentic【理解】

描述的是一个系统所表现出的“ 像 Agent 一样的程度 ”。一个系统越是 Agentic，它就越表现出自主性、目标导向性和主动性。

**Agentic 特性并非凭空出现**，它通过多种具体的工作模式来实现。

![image-20251023220201955](https://bob-blog-image.oss-cn-shanghai.aliyuncs.com/Agent/image-20251023220201955.png)

## 3 Agent五种模式【掌握】

### 3.1 ⼯具使⽤模式（Tool use pattern）

允许 Agent 调用外部工具来弥补自身知识的不足。

agent会自动完成工具的选择和调用，并基于工具调用结果进行最终答案生成。

### 3.2 ReAct 模式 (ReAct Pattern)

将“思考”（Reasoning）和 “行动”（Acting）**紧密地结合在一起，形成一个动态的循环。这个模式让Agent不再是简单地调用工具，而是像人类一样“边想边做”**，从而解决更复杂的问题。

**工作流程：**

1. **思考：** Agent接收用户请求，推理任务需求并制定初步行动计划。
2. **行动：** 根据思考结果，决定并执行具体行动（如调用工具）。
3. **行动输入：** 为选定的工具提供必要参数。
4. **观察：** 接收工具执行结果，作为对环境的“观察”。
5. **循环迭代：** 将观察结果反馈给自己，再次思考并决定下一步，直到达到目标。

### 3.3 反思模式（Reflection pattern）

Agent 在完成一个步骤或整个任务后，对其结果进行评估生成反馈， 然后Agent根据反馈结果进行反思并对结果进行修正。

### 3.4 规划模式（Planning Pattern）

先将一个大目标分解成一个详细的、有序的计划（Plan），然后再逐一执行计划中的每个步骤（每个步骤可能是一个 ReAct 循环）。

### 3.5 多智能体模式 (Multi-agent Pattern)

可以设计多个具有不同角色和能力的 Agent，让它们协同工作来完成极复杂的任务。



### 3.6 Agent 模式的演进关系

上述5种模式构成了一个从简单到复杂的演进阶梯：

**Tool Use (基础)** -\> **ReAct (核心循环)** -\> **Planning (宏观规划)** -\> **Reflection (质量保证)** -\> **Multi-Agent (规模化协作)**

-   **ReAct** 是 **Tool Use** 的规范化和显式化，让工具使用变得有迹可循。
-   **Planning** 是在执行多个 **ReAct** 循环之前的高层战略制定。
-   **Reflection** 是对 **ReAct** 或 **Planning** 执行结果的检查与优化。
-   **Multi-Agent** 是将多个可能使用上述所有模式的 Agent 组织起来，形成一个系统。

通过上述的agent模式的演进过程，它清晰地指明了“如何一步步构建一个更强大的 Agent”。

**TIPS:**

一个真正强大的 Agent 系统，并不会只使用其中一种模式。它会根据任务的复杂性，灵活地将这些模式组合起来。例如，一个 Agent 面对一个复杂问题时，可能会先启动 **规划模式** 来分解任务，然后将子任务交给一个使用  **ReAct 模式** 的执行者，而这个执行者在执行过程中又会调用各种 **工具** ，并在遇到困难时启动 **反思模式** 来修正自己的策略。

这种组合和嵌套的能力，正是 Agentic 系统能够处理现实世界中各种复杂任务的关键。



在项目中的应用：

![image-20251025164352624](https://bob-blog-image.oss-cn-shanghai.aliyuncs.com/Agent/image-20251025164352624.png)



## 4 代码实战【熟悉】

### 4.1 工具使用模式

位置：agent_learn/agent_types/C01_ToolUsePattern.py

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent, create_react_agent
from agent_learn.config import Config

conf = Config()

# 1.创建模型
llm = ChatOpenAI(base_url=conf.base_url,
                 api_key=conf.api_key,
                 model=conf.model_name,
                 temperature=0.1)

# 2.定义工具
@tool
def multiply(a: int, b: int) -> int:
    """用于计算两个整数的乘积。"""
    print(f"正在执行乘法: {a} * {b}")

    return a * b

@tool
def search_weather(city: str) -> str:
    """用于查询指定城市的实时天气。"""
    print(f"正在查询天气: {city}")
    if "北京" in city:
        return "北京今天是晴天，气温25摄氏度。"
    elif "上海" in city:
        return "上海今天是阴天，有小雨，气温22摄氏度。"
    else:
        return f"抱歉，我没有'{city}'的天气信息。"

# 将工具列表放入一个变量
tools = [multiply, search_weather]


# 3.定义一个提示模板，用于控制Agent的思考过程和工具调用
tool_use_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个强大的AI助手，可以访问和使用各种工具来回答问题。请根据用户的问题，决定是否需要调用工具。当需要调用工具时，请使用正确的JSON格式。"),
    ("user", "{input}"),
    ("placeholder", "{agent_scratchpad}") # 这个占位符用于保存 Agent 的思考过程和工具调用历史
])

# 4.创建一个 LLM 能够识别和使用的 Agent
# 使用 create_tool_calling_agent 函数，它能让 LLM 自动判断何时以及如何调用工具
tool_calling_agent = create_tool_calling_agent(llm, tools, tool_use_prompt)

# 5.创建 Agent Executor
# AgentExecutor 负责 Agent 和工具之间的协调
tool_use_executor = AgentExecutor(
    agent=tool_calling_agent,
    tools=tools,
    verbose=True  # 开启 verbose 模式，可以打印详细的执行过程
)

# 6.通用的执行函数，用于运行agent并打印结果
def run_agent_and_print(agent_executor, query):
    """一个通用函数，用于运行Agent并打印结果。"""
    print(f"--- 运行Agent，查询: {query} ---")
    response = agent_executor.invoke({"input": query})
    print(f"\n--- Agent响应: ---")
    print(response.get("output", "没有找到输出。"))
    print("-" * 30 + "\n")


if __name__ == "__main__":
    run_agent_and_print(tool_use_executor, "上海今天的天气怎么样？")
    run_agent_and_print(tool_use_executor, "30乘以5等于多少？ 上海天气怎么样")
```



### 4.2 ReAct模式

注意点：

（1）修改成react模式的提示词

（2）创建react风格的智能体

```python
react_agent = create_react_agent(llm, tools, react_prompt)
```



```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from agent_learn.config import Config

conf = Config()

# 1.创建模型
llm = ChatOpenAI(base_url=conf.base_url,
                 api_key=conf.api_key,
                 model=conf.model_name,
                 temperature=0.1)

# 2.定义工具
# 关键修改：重写 multiply 工具，使其只接受一个字符串参数，并在内部解析。
@tool
def multiply(numbers_str: str) -> int:
    """用于计算两个整数的乘积。

    参数:
        numbers_str (str): 包含两个整数的字符串，用逗号分隔，例如："100,25"。
    返回:
        int: 两个整数的乘积。
    """
    print(f"正在执行乘法: {numbers_str}")
    try:
        a_str, b_str = numbers_str.split(',')
        a = int(a_str.strip())
        b = int(b_str.strip())
        return a * b
    except ValueError:
        return "输入的格式不正确，请确保是两个用逗号分隔的整数，例如：'100,25'"

@tool
def search_weather(city: str) -> str:
    """用于查询指定城市的实时天气。"""
    print(f"正在查询天气: {city}")
    if "北京" in city:
        return "北京今天是晴天，气温25摄氏度。"
    elif "上海" in city:
        return "上海今天是阴天，有小雨，气温22摄氏度。"
    else:
        return f"抱歉，我没有'{city}'的天气信息。"

# 将工具列表放入一个变量
tools = [multiply, search_weather]

# 3. 创建 react模式的 提示词
react_prompt_template = """你是一个有用的 AI 助手，可以访问以下工具：

{tools}

请根据用户输入一步步推理，并按以下规则操作：
1. 每次输出只能包含一个动作（Action 和 Action Input）或一个最终答案（Final Answer）。
2. 如果用户输入包含多个任务，依次处理每个任务，不要一次性输出所有步骤。
3. 每次行动前，说明你的思考（Thought），并选择合适的工具或直接给出最终答案。
4. 如果需要使用工具，格式必须为：
   Thought: [你的思考]
   Action: [工具名称]
   Action Input: [工具的输入参数，例如对于multiply工具，使用'100,25'格式]
5. 如果可以直接回答或所有任务都完成，格式为：
   Thought: [你的思考]
   Final Answer: [最终答案]

可用的工具名称有: {tool_names}

用户输入: {input}
agent的推理过程： {agent_scratchpad}
"""

react_prompt = ChatPromptTemplate.from_template(react_prompt_template)


# 4.创建 ReAct 风格的 Agent
react_agent = create_react_agent(llm, tools, react_prompt)

# 5.创建 Agent Executor
react_executor = AgentExecutor(
    agent=react_agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True  # 启用错误处理，自动重试解析错误
)


if __name__ == '__main__':
    # 单问题
    # print(react_executor.invoke({"input": "请计算100乘以25"}))

    # 多问题
    print(react_executor.invoke({"input": "请计算100乘以25，并查询上海的天气"}))
```



### 4.3 反思模式

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from agent_learn.config import Config

conf = Config()

# 1.创建模型
llm = ChatOpenAI(base_url=conf.base_url,
                 api_key=conf.api_key,
                 model=conf.model_name,
                 temperature=0.1)

# 2.初始响应 Prompt: 用于生成第一次的回答
initial_response_prompt = ChatPromptTemplate.from_template(
    "请根据以下问题给出你的初步回答: {question}"
)
initial_response_chain = initial_response_prompt | llm | StrOutputParser()


# 3.反思 Prompt: 用于接收用户反馈并优化回答
reflection_prompt = ChatPromptTemplate.from_template(
    """你是一个专业的、善于反思的AI助手。你之前给出了以下回答：
---
{previous_response}
---
现在，你收到了用户对你的回答给出的反馈：
---
{user_feedback}
---
请根据用户的反馈，认真反思你之前的回答，并生成一个更准确、更完善的新回答。
新回答:"""
)
reflection_chain = reflection_prompt | llm | StrOutputParser()


# 5.模拟反射过程
def reflect_and_refine(query: str, feedback: str):
    """模拟一个完整的反射过程，从初始响应到优化后的响应。"""

    print("--- 启动反射模式 ---")
    print(f"用户查询: {query}")

    # LLM 生成初步响应
    print("\n生成初步响应...")
    initial_response = initial_response_chain.invoke({"question": query})
    print(f"LLM 初步响应:\n{initial_response}")

    # 这里是模拟用户的反馈，在实际的项目中，可以让 自身大模型/专用的评估模型 进行评价；或者是调用工具进行评估；还是纯人工的反馈
    print("\n模拟用户反馈...")
    print(feedback)

    # LLM 反思并生成新回答
    print("\nLLM 反思并生成新回答...")
    refined_response = reflection_chain.invoke(
        {"previous_response": initial_response, "user_feedback": feedback}
    )

    # 返回结果
    print(f"LLM 优化后的回答:\n{refined_response}")
    return refined_response


# 6.运行并测试
if __name__ == "__main__":
    # 模拟用户查询
    initial_question = "请用一句话介绍一下 LangChain。"
    # 模拟用户反馈，指出初步回答的不足
    user_feedback_text = "你的回答太简单了，请更详细地解释一下 LangChain 的核心概念，比如 Agent 和 Chain 的区别。"
    # 运行反射过程
    reflect_and_refine(initial_question, user_feedback_text)

```

### 4.4 规划模式

这个案例分了两个智能体，一个做任务规划，一个做任务执行。因为有多个智能体，所以也可以称为多智能体模式。

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import AgentExecutor, create_react_agent
from agent_learn.config import Config

conf = Config()

# 1.创建模型
llm = ChatOpenAI(base_url=conf.base_url,
                 api_key=conf.api_key,
                 model=conf.model_name,
                 temperature=0.1)


# 2.定义工具
@tool
def multiply(numbers_str: str) -> int:
    """用于计算两个整数的乘积。

    参数:
        numbers_str (str): 包含两个整数的字符串，用逗号分隔，例如："100,25"。
    返回:
        int: 两个整数的乘积。
    """
    print(f"正在执行乘法: {numbers_str}")
    try:
        a_str, b_str = numbers_str.split(',')
        a = int(a_str.strip())
        b = int(b_str.strip())
        return a * b
    except ValueError:
        return "输入的格式不正确，请确保是两个用逗号分隔的整数，例如：'100,25'"


@tool
def search_weather(city: str) -> str:
    """用于查询指定城市的实时天气。"""
    print(f"正在查询天气: {city}")
    if "北京" in city:
        return "北京今天是晴天，气温25摄氏度。"
    elif "上海" in city:
        return "上海今天是阴天，有小雨，气温22摄氏度。"
    else:
        return f"抱歉，我没有'{city}'的天气信息。"


# 将工具列表放入一个变量
tools = [multiply, search_weather]

# 3.定义规划器 (Planner) 和执行者 (Executor) 的 Prompt
# 3.1 规划器的 Prompt
# 规划器的职责是分析用户任务，并将其分解成一系列简单的、可执行的子任务。
planner_prompt = ChatPromptTemplate.from_template(
    """你是一个任务规划师，你的工作是将用户提出的一个复杂任务分解成一系列清晰、可执行的步骤。
    你的输出应该是一个简单的任务列表，每行一个任务。

    例子:
    用户任务: "请先查上海的天气，然后计算20乘以30。"
    任务列表:
    - 查找上海的天气
    - 计算20乘以30的结果

    用户任务: {user_input}
    任务列表:
    """
)
# 规划器链，它只负责生成文本化的任务列表
planner_chain = planner_prompt | llm | StrOutputParser()

# 3.2 执行者的 Prompt
# 执行者的职责是执行单个任务。在这里，我们使用 ReAct 模式作为执行者，因为它能根据任务描述选择并调用正确的工具。
# 注意：我们使用一个简化版的 ReAct Prompt，因为它只需要处理单个任务。
executor_react_prompt_template = """你是一个专业的工具执行者，可以访问以下工具：

{tools}

根据你的思考（Thought）决定下一步的行动（Action）。你的行动必须遵循以下格式：
Thought: 我需要思考如何完成任务。
Action: [工具名称]
Action Input: [工具的输入参数，对于multiply工具，请使用'100,25'这样的格式]

可用的工具名称有: {tool_names}

当你获取了所有必要信息并可以给出最终答案时，请以以下格式结束：
Thought: 我已经有了最终答案。
Final Answer: [最终答案]

请执行以下任务：
{input}
{agent_scratchpad}
"""
executor_prompt = ChatPromptTemplate.from_template(executor_react_prompt_template)

# 4.创建 ReAct Agent 作为执行者
executor_agent = create_react_agent(llm, tools, executor_prompt)
executor_executor = AgentExecutor(
    agent=executor_agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True  # 启用错误处理，自动重试解析错误
)

# 5.定义并运行规划模式的工作流
def execute_planning_pattern(query: str):
    print("--- 启动规划模式 ---")

    # 规划器分解任务
    print("\n规划器正在分解任务...")
    plan = planner_chain.invoke({"user_input": query})
    tasks = [task.strip() for task in plan.split('\n') if task.strip()]
    print("规划器生成的任务列表:")
    for i, task in enumerate(tasks):
        print(f"  {i + 1}. {task}")

    # 执行者逐一执行任务
    print("\n执行者正在逐一执行任务...")
    for i, task in enumerate(tasks):
        print(f"\n--- 执行任务 {i + 1}: {task} ---")
        result = executor_executor.invoke({"input": task})
        print(f"执行者输出: {result}")

    print("\n--- 所有任务执行完毕！---")


if __name__ == "__main__":
    test_query = "请先计算 50 乘以 60 的结果，然后告诉我上海的天气怎么样？"
    execute_planning_pattern(test_query)
```



### 4.5 多智能体模式

这个案例分了3个智能体，一个是计算专家用来处理计算任务；一个是信息专家，用来处理信息检索任务；还有一个内容整合智能体，用于生成最终答案。

这里案例里少了一个任务规划的智能体，这部分工作是由我们手动来完成的。

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, \
    HumanMessagePromptTemplate  # 导入所有必需的 Prompt 类
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.output_parsers import StrOutputParser
from agent_learn.config import Config

conf = Config()

# 1.创建模型
llm = ChatOpenAI(base_url=conf.base_url,
                 api_key=conf.api_key,
                 model=conf.model_name,
                 temperature=0.1)


# 2.创建数学计算的智能体
# 2.1 计算工具
@tool
def multiply(a: int, b: int) -> int:
    """用于计算两个整数的乘积。

    参数:
        a (int): 第一个整数。
        b (int): 第二个整数。
    """
    print(f"\n[计算专家] -> 正在执行乘法: {a} * {b}")
    return a * b


@tool
def add(a: int, b: int) -> int:
    """用于计算两个整数的和。

    参数:
        a (int): 第一个整数。
        b (int): 第二个整数。
    """
    print(f"\n[计算专家] -> 正在执行加法: {a} + {b}")
    return a + b

# 2.2 创建“计算专家” Agent
math_tools = [multiply, add]
# 创建完整的 Tool Calling Prompt
# 这包括一个系统消息，一个用户消息占位符，以及一个 Agent 中间思考过程的占位符。
math_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("你是一个强大的数学计算专家，可以访问和使用各种数学工具。"),
    HumanMessagePromptTemplate.from_template("{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])
# 创建 math_Agent
math_agent = create_tool_calling_agent(llm, math_tools, math_prompt)
# 创建 math Agent Executor
math_executor = AgentExecutor(
    agent=math_agent,
    tools=math_tools,
    verbose=True
)

# 3.创建信息检索的智能体
# 3.1 信息查询工具
@tool
def search_weather(city: str) -> str:
    """用于查询指定城市的实时天气。"""
    print(f"正在查询天气: {city}")
    if "北京" in city:
        return "北京今天是晴天，气温25摄氏度。"
    elif "上海" in city:
        return "上海今天是阴天，有小雨，气温22摄氏度。"
    else:
        return f"抱歉，我没有'{city}'的天气信息。"


@tool
def get_current_date() -> str:
    """用于获取当前日期。"""
    print("\n[信息专家] -> 正在获取当前日期...")
    import datetime
    return datetime.date.today().strftime("%Y年%m月%d日")


# 3.2 创建“信息专家” Agent
info_tools = [search_weather, get_current_date]
# 手动创建完整的 Tool Calling Prompt
info_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("你是一个强大的信息查询专家，可以访问和使用各种查询工具。"),
    HumanMessagePromptTemplate.from_template("{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])
# 创建 info Agent
info_agent = create_tool_calling_agent(llm, info_tools, info_prompt)
# 创建 info Agent Executor
info_executor = AgentExecutor(
    agent=info_agent,
    tools=info_tools,
    verbose=True
)


# 4.协调和总结工作流
def multi_agent_workflow(query: str, math_task: str, info_task: str):
    print("--- 启动多智能体协作流程 ---")
    print(f"\n用户原始请求: {query}")

    # 4.1 让“计算专家”执行任务
    print("\n[主程序] -> 将任务分配给计算专家...")
    math_result = math_executor.invoke({"input": math_task}).get("output")
    print(f"\n[主程序] -> 计算专家返回结果: {math_result}")

    # 4.2 让“信息专家”执行任务
    print("\n[主程序] -> 将任务分配给信息专家...")
    info_result = info_executor.invoke({"input": info_task}).get("output")
    print(f"\n[主程序] -> 信息专家返回结果: {info_result}")

    # 4.3 使用 LLM 进行最终结果总结
    print("\n[主程序] -> 使用大模型进行最终总结...")
    summarize_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个善于总结和整合信息的助手。请根据以下信息，为用户原始请求生成一个完整的回答。"),
        ("human",
         f"用户请求: {query}\n\n计算结果: {math_result}\n\n信息查询结果: {info_result}\n\n请整合以上信息，生成一个连贯的最终回答。")
    ])
    summarize_chain = summarize_prompt | llm | StrOutputParser()
    final_answer = summarize_chain.invoke({"query": query})

    print("\n--- 协作流程已完成！---")
    print(f"最终综合回答:\n{final_answer}")
    return final_answer


if __name__ == "__main__":
    # 定义用户原始请求和分配给每个Agent的子任务
    original_query = "请先计算 25 乘以 4，然后告诉我北京今天的天气和当前日期。"
    math_task = "计算 25 乘以 4"
    info_task = "查询北京今天的天气和当前日期"

    # 启动工作流
    multi_agent_workflow(original_query, math_task, info_task)
```



# 四、A2A协议

## 1 Agent2Agent Protocol【理解】

A2A协议就是不同智能体进行沟通协作的协议。



作用：**安全协作（可以保证agent之间的信息是安全的）**、**任务与状态管理（提交一个任务后，可以跟踪任务的状态和处理结果）**、**用户体验协商（智能体可以根据用户的问题和反馈进行调整，提高用户的体验）**、**能力发现（agent通过AgentCard 来展示自己的功能，其他agent可以自动读取AgentCard 中的信息来了解该智能体的功能）**



### 1.1 Agent2Agent 架构剖析

A2A核心角色：

- User：用户是协议中的关键主体，主要负责进行认证和授权操作，确保交互的安全性和合法性。
- Client Agent：客户端 Agent 是任务的发起者，它代表用户提出需求或请求。
- Server Agent：服务端 Agent 是任务的执行者，它接收来自客户端 Agent 的请求，并执行相应的操作。



------

### 1.2 Agent2Agent 核心概念

**AgentSkill**：AgentSkill 定义了单个智能体（Agent）所具备的、可被外部调用的具体功能或能力。

**AgentCard**：AgentCard 是描述一个智能体身份、能力（AgentSkill）、接口信息和元数据的标准化声明文件，用于代理发现和服务注册。

**Task**：Task 指的是具体的需要完成目标，会包含关于session_id、状态、任务的内容、处理结果等信息。

**TaskState**：TaskState是任务状态枚举类，定义了任务的可能状态，包括SUBMITTED/COMPLETED等。

**TaskStatus**：TaskStatus 表示 A2A 任务的当前状态对象，包括状态枚举（TaskState）、附加消息和时间戳。

**A2AServer**：A2AServer是A2A协议的核心实现类，用于 构建代理服务器 。

**artifacts**：artifacts 是 A2A 协议中 Task 对象的核心字段之一，用于存储任务执行后的输出产物（结果）。

**AgentNetwork**：AgentNetwork 是 A2A 协议中的agent网络管理类，用于集中管理和发现 A2AServer。

**AIAgentRouter**：AIAgentRouter 是负责根据任务需求和 AgentCard 信息，将任务路由到最合适智能体的组件。



## 2 A2AServer结合MCP Server【掌握】

接下来通过一个案例，来看一下如何将A2AServer和MCP Server结合起来进行使用。

**调用逻辑：**

![image-20251026105504578](https://bob-blog-image.oss-cn-shanghai.aliyuncs.com/Agent/image-20251026105504578.png)



### 2.1 创建MCP Server

位置：agent_learn/A2A_base/a2a_mcp_collaboration/mcp_weather_tool_agent.py

```python
import uvicorn
from python_a2a.mcp import FastMCP, create_fastapi_app

mcp = FastMCP(name="WeatherTool")

@mcp.tool(name="get_weather", description="获取城市天气")
def get_weather(city: str) -> str:
    print(f"[MCP 工具 Agent 日志] 收到工具调用，查询城市: {city}")
    if city == "北京":
        return "北京今天阳光明媚，29°C"
    return f"找不到 {city} 的天气"


if __name__ == "__main__":
    app = create_fastapi_app(mcp)
    print("[MCP 工具 Agent] 已启动，在 http://127.0.0.1:6005")
    uvicorn.run(app, host="127.0.0.1", port=6005)
```



### 2.2 创建A2AServer

位置：agent_learn/A2A_base/a2a_mcp_collaboration/a2a_main_agent.py

```python
from python_a2a import A2AServer, run_server, AgentCard, AgentSkill, TaskStatus, TaskState
from python_a2a.mcp import MCPClient
import asyncio

# A2A Agent 的名片
agent_card = AgentCard(
    name="WeatherServer",
    description="用来查询天气",
    url="http://127.0.0.1:8005",
    skills=[AgentSkill(name="查询天气", description="查询指定城市的天气")]
)

class WeatherServer(A2AServer):
    def __init__(self):
        super().__init__(agent_card=agent_card)
        # 创建一个MCPClient对象
        self.mcp_client = MCPClient('http://127.0.0.1:6005')

    def handle_task(self, task):
        print("收到A2A任务的task:=>", task)
        # 获取任务内容
        query = (task.message or {}).get('content', {}).get('text', '')

        # 具体处理逻辑
        if "天气" in query:
            # 获取城市
            city = "北京"
            # 调用MCP Server中的工具
            # 因为这个工具是异步的，所以需要使用 asyncio.run() 进行调用，否则只会返回一个调用的对象！！
            weather_result = asyncio.run(self.mcp_client.call_tool(tool_name="get_weather", city=city))
            print("天气查询结果:=>", weather_result)
            # 将查询结果放到 task.artifacts
            task.artifacts = [{"parts": [{"type": "text", "text": weather_result}]}]
        else:
            task.artifacts = [{"parts": [{"type": "text", "text": "无法理解的任务"}]}]

        # 设置任务状态为完成
        task.status = TaskStatus(TaskState.COMPLETED)

        print(f"[{self.agent_card.name} 日志] 任务完成，结果已返回给 A2A")
        print("task:=>",task)
        print("task.artifacts:=>",task.artifacts)

        return task


if __name__ == '__main__':
    run_server(WeatherServer(), host="127.0.0.1", port=8005)
```



### 2.3 客户端

位置：agent_learn/A2A_base/a2a_mcp_collaboration/main_client.py

```python
import asyncio
from python_a2a import A2AClient

async def main():
    # 客户端只知道主控 Agent 的存在
    magent_client = A2AClient("http://127.0.0.1:8005")
    print("[主客户端日志] 准备向主控 Agent 发送任务...")

    # 发起 A2A 调用
    query = "请帮我查一下北京的天气"
    result = magent_client.ask(query)
    print(f"[主客户端日志] 收到最终结果: '{result}'")


if __name__ == "__main__":
    # 请确保 mcp_weather_tool_agent.py 和 a2a_main_agent.py 正在运行...
    asyncio.run(main())
```



## 3 A2AServer串行【掌握】

本部分主要实现A2A串行协作的场景。

### 3.1 weather_agent

位置：agent_learn/A2A_base/a2a_serial/weather_agent.py

```python
from python_a2a import A2AServer, run_server, AgentCard, AgentSkill, TaskStatus, TaskState

# A2A Agent 的名片
agent_card = AgentCard(
    name="WeatherAgentServer",
    description="一个天气预报查询的专家 Agent",
    url="http://127.0.0.1:5008",
    skills=[AgentSkill(name="query", description="接受天气查询查询",examples=["天气北京"])]
)


class WeatherAgentServer(A2AServer):
    def __init__(self):
        super().__init__(agent_card=agent_card)

    def handle_task(self, task):
        print("收到A2A任务的task:=>", task)
        query = (task.message or {}).get("content", {}).get("text", "")
        print(f"[{self.agent_card.name} 日志] 收到 A2A 任务: '{query}'")
        # 决策：如果查询包含“天气”，就调用 MCP 工具
        if "天气" in query:
            print(f"[{self.agent_card.name} 日志] 决策：任务需要天气数据，准备调用工具...")
            try:
                # 这里的结果可以来自于 MCP 模块，这里我们直接模拟结果
                weather_result = {"温度": 30, "天气": "晴天"}
                print(f"[{self.agent_card.name} 日志] 从 MCP 工具获得结果: '{weather_result}'")
                # 将结果保存为任务 artifacts,artifacts是任务的输出结果
                task.artifacts = [{"parts": [{"type": "text", "text": weather_result}]}]
            except Exception as e:
                error_msg = f"调用 工具失败: {e}"
                print(f"[{self.agent_card.name} 日志] {error_msg}")
                task.artifacts = [{"parts": [{"type": "text", "text": error_msg}]}]
        else:
            task.artifacts = [{"parts": [{"type": "text", "text": "无法理解的任务"}]}]

        task.status = TaskStatus(state=TaskState.COMPLETED)
        print(f"[{self.agent_card.name} 日志] 任务处理完毕")
        print(f"[{self.agent_card.name} 日志] 输出结果task: {task}")
        print(f"[{self.agent_card.name} 日志] 输出结果task.artifacts: {task.artifacts}")
        return task


if __name__ == "__main__":
    server = WeatherAgentServer()
    print(f"[{server.agent_card.name}] 已启动，在 {server.agent_card.url}")
    run_server(server, host="127.0.0.1", port=5008)
```



### 3.2 ticket_agent

位置：agent_learn/A2A_base/a2a_serial/ticket_agent.py

```python
from python_a2a import A2AServer, run_server, AgentCard, AgentSkill, TaskStatus, TaskState

ticket_card = AgentCard(
    name="TicketAgentServer",
    description="一个可以预订票务的专家 Agent。",
    url="http://127.0.0.1:5009",
    version="1.0.0",
    skills=[AgentSkill(name="book_ticket", description="预订票务")]
)

class TicketServer(A2AServer):
    def __init__(self):
        super().__init__(agent_card=ticket_card)

    def handle_task(self, task):
        print("收到A2A任务的task:=>", task)
        query = (task.message or {}).get("content", {}).get("text", "")
        print(f"[{self.agent_card.name} 日志] 收到 A2A 任务: '{query}'")

        if "上海" in query and "北京" in query:
        # 这里的结果可以来自于 MCP 模块，这里我们直接模拟结果
            train_result = "上海到北京的火车票已经预订成功！  G1001,10车1A "
        else:
            train_result = "请输入明确的出发地和目的地。"

        # 返回处理结果，并更新状态
        print(f"[{self.agent_card.name} 日志] 返回结果: {train_result}")
        task.artifacts = [{"parts": [{"type": "text", "text": train_result}]}]
        task.status = TaskStatus(state=TaskState.COMPLETED)
        print(f"[{self.agent_card.name} 日志] 任务处理完毕")
        print(f"[{self.agent_card.name} 日志] 输出结果task: {task}")
        print(f"[{self.agent_card.name} 日志] 输出结果task.artifacts: {task.artifacts}")

        return task


if __name__ == "__main__":
    server = TicketServer()
    print(f"[{server.agent_card.name}] 启动成功，服务地址: {server.agent_card.url}")
    run_server(server, host="127.0.0.1", port=5009)
```



### 3.3 main_orchestrator

位置：agent_learn/A2A_base/a2a_serial/main_orchestrator.py

```python
import asyncio
from python_a2a import AgentNetwork, A2AClient, Task, Message, MessageRole, TextContent
import json
import uuid
from time import sleep


async def main():
    # 步骤1：初始化 AgentNetwork 并注册专家 Agent
    # 1.1 创建 AgentNetwork 实例，用于管理 Agent 集合
    network = AgentNetwork(name="TravelOrchestrator")
    # 1.2 添加票务 Agent
    network.add("TicketAgent", "http://127.0.0.1:5009")
    # 1.3 添加天气 Agent
    network.add("WeatherAgent", "http://127.0.0.1:5008")
    print("[主控日志] AgentNetwork 初始化完成，已添加专家代理：")
    for agent_info in network.list_agents():
        print(json.dumps(agent_info, indent=4, ensure_ascii=False))
    print("-" * 50)

    # 步骤2：执行串行任务流
    # 2.1 任务一：查询天气
    weather_query = "北京的天气怎么样"  # 用户请求

    #  获取 WeatherAgent 的客户端
    weather_client = network.get_agent("WeatherAgent")

    # 调用客户端
    # 方式一：直接调用
    # weather_result = weather_client.ask(weather_query)
    # print("[主控日志] 天气查询结果：")
    # print(weather_result)

    # 方式二：异步调用
    # Message用来存储具体的任务内容
    message = Message(content=TextContent(text=weather_query), role=MessageRole.USER)
    # Task中存储和封装Message
    weather_task = Task(message=message.to_dict(), id="task-" + str(uuid.uuid4()))
    # 使用 send_task_async 实现异步调用
    weather_result =  await weather_client.send_task_async(weather_task)

    # 解析结果
    weather_info = "未知天气"
    try:
        # 获取 artifacts 中的文本部分
        weather_parts = weather_result.artifacts[0]["parts"]
        if weather_parts and weather_parts[0].get("type") == "text":
            weather_info = weather_parts[0].get("text")
            print(f"[主控日志] 收到 WeatherAgent 的结果: '{weather_info}'")
    except Exception as e:
        print(f"[主控日志] 解析天气结果出错: {e}")


    # 2.2 任务二： 根据天气的结果预定火车票
    ticket_query = f"预订一张从北京到上海的火车票，当前天气是：{weather_info}"

    #  获取 TicketAgent 的客户端
    ticket_client = network.get_agent("TicketAgent")

    # 异步调用
    # Message用来存储具体的任务内容
    ticket_message = Message(content=TextContent(text=ticket_query), role=MessageRole.USER)
    # Task中存储和封装Message
    ticket_task = Task(message=ticket_message.to_dict(), id="task-" + str(uuid.uuid4()))
    # 使用 send_task_async 实现异步调用
    ticket_result =  await ticket_client.send_task_async(ticket_task)

    # 打印最终结果
    print(f"\n[主控日志] 收到 TicketAgent 的最终结果：")
    print(json.dumps(ticket_result.to_dict(), indent=4, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
```



## 4 A2A实战案例【掌握】

A2A（Agent-to-Agent）协议是一种支持代理间通信的框架，允许不同 AI 代理通过任务、消息和产物进行协作。在本实战中，我们基于四个脚本（router_A2Aagent_Server.py、weather_agent.py、ticket_agent.py 和 main.py），实现一个 LLM 驱动的路由系统：

- 路由服务器（router_A2Aagent_Server.py）：使用 LangChain 和 OpenAI LLM 作为路由决策引擎。
- 票务代理（TicketAgentServer）：处理火车票预订请求。
- 天气代理（WeatherAgentServer）：处理天气查询请求。
- 主控客户端（main.py）：使用 AgentNetwork 和 AIAgentRouter 路由查询到合适代理，获取结果。

**目标:**

通过4个脚本，基于A2A 协议的路由协作，LLM 识别意图，将查询路由到票务或天气代理，模拟旅行助手系统。实战通过异步运行服务器和客户端，验证意图识别和任务处理。

### 4.1 router_agent

**实现目标**：router_A2Aagent_Server主要构建 LLM 路由代理服务器进行意图识别和路由决策。使用 LangChain 转换 OpenAI LLM 为 A2A 服务器，监听端口 5555，支持异步路由决策。

**核心功能**：

- 使用 ChatOpenAI 创建 LLM。
- 通过 to_a2a_server(llm) 转换为 A2A 服务器。
- 异步启动 run_server，提供路由服务。

**代码位置**：agent_learn/A2A_base/a2a_case/router_A2Aagent_Server.py

```python
from langchain_openai import ChatOpenAI
from python_a2a import run_server
from python_a2a.langchain import to_a2a_server
import asyncio
from agent_learn.config import Config

conf = Config()

async def main():
    # 创建LangChain LLM
    llm = ChatOpenAI(base_url=conf.base_url,
                     api_key=conf.api_key,
                     model=conf.model_name,
                     temperature=0.1,
                     streaming=True)
    # 转换为A2A服务器
    llm_server = to_a2a_server(llm)
    print(llm_server.agent_card)
    # 启动服务器
    run_server(llm_server, port=5555)

if __name__ == '__main__':
    asyncio.run(main())
```

### 4.2 ticket_agent

**代码位置**：agent_learn/A2A_base/a2a_case/ticket_agent.py

```python
from python_a2a import A2AServer, run_server, AgentCard, AgentSkill, TaskStatus, TaskState

ticket_card = AgentCard(
    name="TicketAgentServer",
    description="一个可以预订票务的专家 Agent。",
    url="http://127.0.0.1:5009",
    version="1.0.0",
    skills=[AgentSkill(name="book_ticket", description="预订票务")]
)

class TicketServer(A2AServer):
    def __init__(self):
        super().__init__(agent_card=ticket_card)

    def handle_task(self, task):
        print("收到A2A任务的task:=>", task)
        query = (task.message or {}).get("content", {}).get("text", "")
        print(f"[{self.agent_card.name} 日志] 收到 A2A 任务: '{query}'")

        if "上海" in query and "北京" in query:
        # 这里的结果可以来自于 MCP 模块，这里我们直接模拟结果
            train_result = "上海到北京的火车票已经预订成功！  G1001,10车1A "
        else:
            train_result = "请输入明确的出发地和目的地。"

        # 返回处理结果，并更新状态
        print(f"[{self.agent_card.name} 日志] 返回结果: {train_result}")
        task.artifacts = [{"parts": [{"type": "text", "text": train_result}]}]
        task.status = TaskStatus(state=TaskState.COMPLETED)
        print(f"[{self.agent_card.name} 日志] 任务处理完毕")
        print(f"[{self.agent_card.name} 日志] 输出结果task: {task}")
        print(f"[{self.agent_card.name} 日志] 输出结果task.artifacts: {task.artifacts}")

        return task


if __name__ == "__main__":
    server = TicketServer()
    print(f"[{server.agent_card.name}] 启动成功，服务地址: {server.agent_card.url}")
    run_server(server, host="127.0.0.1", port=5009)
```



### 4.3 weather_agent

**代码位置**：agent_learn/A2A_base/a2a_case/weather_agent.py

```python
from python_a2a import A2AServer, run_server, AgentCard, AgentSkill, TaskStatus, TaskState

# A2A Agent 的名片
agent_card = AgentCard(
    name="WeatherAgentServer",
    description="一个天气预报查询的专家 Agent",
    url="http://127.0.0.1:5008",
    skills=[AgentSkill(name="query", description="接受天气查询查询",examples=["天气北京"])]
)


class WeatherAgentServer(A2AServer):
    def __init__(self):
        super().__init__(agent_card=agent_card)

    def handle_task(self, task):
        print("收到A2A任务的task:=>", task)
        query = (task.message or {}).get("content", {}).get("text", "")
        print(f"[{self.agent_card.name} 日志] 收到 A2A 任务: '{query}'")
        # 决策：如果查询包含“天气”，就调用 MCP 工具
        if "天气" in query:
            print(f"[{self.agent_card.name} 日志] 决策：任务需要天气数据，准备调用工具...")
            try:
                # 这里的结果可以来自于 MCP 模块，这里我们直接模拟结果
                weather_result = {"温度": 30, "天气": "晴天"}
                print(f"[{self.agent_card.name} 日志] 从 MCP 工具获得结果: '{weather_result}'")
                # 将结果保存为任务 artifacts,artifacts是任务的输出结果
                task.artifacts = [{"parts": [{"type": "text", "text": weather_result}]}]
            except Exception as e:
                error_msg = f"调用 工具失败: {e}"
                print(f"[{self.agent_card.name} 日志] {error_msg}")
                task.artifacts = [{"parts": [{"type": "text", "text": error_msg}]}]
        else:
            task.artifacts = [{"parts": [{"type": "text", "text": "无法理解的任务"}]}]

        task.status = TaskStatus(state=TaskState.COMPLETED)
        print(f"[{self.agent_card.name} 日志] 任务处理完毕")
        print(f"[{self.agent_card.name} 日志] 输出结果task: {task}")
        print(f"[{self.agent_card.name} 日志] 输出结果task.artifacts: {task.artifacts}")
        return task


if __name__ == "__main__":
    server = WeatherAgentServer()
    print(f"[{server.agent_card.name}] 已启动，在 {server.agent_card.url}")
    run_server(server, host="127.0.0.1", port=5008)
```



### 4.4 main

**实现目标：**作为主控客户端，使用 AgentNetwork 和 AIAgentRouter 路由查询到票务或天气代理，获取完整响应。

**核心功能：**

- 初始化 AgentNetwork，添加代理 URL。
- 使用 AIAgentRouter 路由查询，返回代理名称和置信度。
- 发送 Task 到选择的代理，解析 artifacts 中的 parts。

**代码位置**：

```python
import asyncio
from python_a2a import AgentNetwork, AIAgentRouter, A2AClient, Task, Message, MessageRole, TextContent
import json
import uuid

from time import sleep


async def main():
    # 1.创建 AgentNetwork
    # 实例化对象
    network = AgentNetwork(name='TravelAgentNetwork')
    # 添加 agent  name可以自己定义
    network.add(name='WeatherAgent', agent_or_url='http://127.0.0.1:5008')
    network.add(name='TicketAgent', agent_or_url='http://127.0.0.1:5009')
    # 打印agent server信息
    for agent_info in network.list_agents():
        print(json.dumps(agent_info, indent=4, ensure_ascii=False))

    # 2.创建AIAgentRouter
    # 创建路由器
    router = AIAgentRouter(llm_client=A2AClient("http://127.0.0.1:5555"),
                           agent_network=network)

    # 3. 测试查询
    queries = [
        "帮我查下北京的天气",  # 应该路由到 WeatherAgent
        "预订一张从北京到上海的火车票"  # 应该路由到 TicketAgent
    ]
    for query in queries:
        print(f"[主控日志] 用户查询: '{query}'")
        # 使用路由器选择agent
        agent_name, confidence = router.route_query(query)
        print(f"[主控日志] 匹配的agent: '{agent_name}', 匹配度: {confidence}")
        if agent_name:
            # 基于 agent_name 获取对应的 agent server
            agent_client = network.get_agent(agent_name)
            if agent_client:
                # Message用来存储具体的任务内容
                message = Message(content=TextContent(text=query), role=MessageRole.USER)
                # Task中存储和封装Message
                agent_task = Task(message=message.to_dict(), id="task-" + str(uuid.uuid4()))
                # 使用 send_task_async 实现异步调用
                try:
                    agent_result = await agent_client.send_task_async(agent_task)
                    print("[主控日志] agent返回结果:")
                    print(json.dumps(agent_result.to_dict(), indent=4, ensure_ascii=False))
                except Exception as e:
                    print(f"[主控日志] agent调用失败: '{e}'")


if __name__ == '__main__':
    asyncio.run(main())
```



### 4.5 扩展: multi_intents

**核心功能**：演示一个能够处理多意图查询的主控 Agent。

**流程**：用户查询 -> LLM分解子查询 -> 路由到不同的专家Agent -> 并行执行 -> 收集并展示结果。

<img src="https://bob-blog-image.oss-cn-shanghai.aliyuncs.com/Agent/image-20251025222404917.png" alt="image-20251025222404917" style="zoom:67%;" />

==总结：==

- 复杂任务如何拆解？——使用任务拆解agent（大模型，用提示词的方式），将复杂任务拆解成子任务。【同agent的规划模式】
- 如何做到任务的并行？——先将任务的协程对象进行保存，然后调用asyncio.gather()一起执行这些任务。

**代码位置**：agent_learn/A2A_base/a2a_case/multi_intents.py

```python
# 步骤1：导入所需的库和模块
import asyncio  # 导入 asyncio 库，用于实现异步和并发操作
from python_a2a import AgentNetwork, AIAgentRouter, A2AClient, Task, Message, MessageRole, TextContent # 从 python_a2a 库导入 Agent 协作所需的核心类和对象
from langchain_openai import ChatOpenAI  # 导入 LangChain 的 ChatOpenAI，用于与大语言模型交互
from langchain_core.prompts import PromptTemplate  # 导入 LangChain 的 PromptTemplate，用于定义提示模板
from langchain_core.output_parsers import StrOutputParser  # 导入 LangChain 的 StrOutputParser，用于解析 LLM 输出为字符串
import json  # 导入 json 库，用于处理 JSON 格式的数据
import uuid  # 导入 uuid 库，用于生成唯一的任务 ID
from time import sleep  # 导入 sleep 函数，用于模拟处理延迟
from agent_learn.config import Config # 导入自定义的 Config 类，用于加载配置信息
import re  # 导入 re 模块，用于正则表达式处理

# 步骤2：初始化配置和LLM
# 2.1 从配置文件加载配置
conf = Config()

# 2.2 配置 LLM 用于分解查询
decompose_llm = ChatOpenAI(
    model=conf.model_name,
    api_key=conf.api_key,
    base_url=conf.base_url,
    temperature=0.1,
    streaming=True  #启用流式处理
)

# 2.3 定义分解查询的提示模板
decompose_prompt = PromptTemplate.from_template(""" 
将以下用户查询分解为独立的子查询，每个子查询对应一个单一意图。 
返回 JSON 格式的列表：{{"sub_queries": ["子查询1", "子查询2", ...]}}
示例：
查询: "预订票,查天气"
输出: {{"sub_queries": ["预订票", "查天气"]}}
查询: {query}
""")

# 2.4 构建分解链
decompose_chain = decompose_prompt | decompose_llm | StrOutputParser()


# 步骤3：主函数，执行多意图协作流程
async def main():
    # 1）创建 AgentNetwork
    # 实例化对象
    network = AgentNetwork(name='TravelAgentNetwork')
    # 添加 agent  name可以自己定义
    network.add(name='WeatherAgent', agent_or_url='http://127.0.0.1:5008')
    network.add(name='TicketAgent', agent_or_url='http://127.0.0.1:5009')
    # 打印agent server信息
    for agent_info in network.list_agents():
        print(json.dumps(agent_info, indent=4, ensure_ascii=False))

    # 2）创建AIAgentRouter
    # 创建路由器
    router = AIAgentRouter(llm_client=A2AClient("http://127.0.0.1:5555"),
                           agent_network=network)

    # 3）测试查询
    queries = [
        "帮我查下北京的天气，并预订一张从北京到上海的火车票", # 复合意图
        "帮我查下北京的天气",  # 应该路由到 WeatherAgent
        "预订一张从北京到上海的火车票",  # 应该路由到 TicketAgent
    ]
    for query in queries:
        print(f"[主控日志] 用户查询: '{query}'")
        try:
            # 3.1）使用 LLM 分解查询为子查询
            decompose_result = decompose_chain.invoke({"query": query})
            print(f"[主控日志] 分解结果: {decompose_result}")

            # 使用正则表达式清理LLM输出中的JSON标记
            decompose_response = re.sub(r'^```json\n|\n```$', '', decompose_result.strip())
            decompose_data = json.loads(decompose_response)
            # 从JSON中获取子查询列表，如果失败则使用原始查询
            sub_queries = decompose_data.get("sub_queries", [query])
        except Exception as e:
            print(f"[主控日志] 分解失败: {e}")
            sub_queries = [query]  # 发生错误时，将原始查询作为唯一的子查询
        print(f"[主控日志] 子查询列表: {sub_queries}")

        # 3.2）处理每一个子查询，获取对应的处理任务，后续再一起执行！
        tasks = []  # 创建一个空列表，用于存放所有要并行执行的异步任务
        agent_names = []  # 创建一个空列表，用于记录每个任务对应的Agent名称
        confidences = []  # 创建一个空列表，用于记录路由的置信度
        for sub_query in sub_queries:  # 遍历所有分解出的子查询
            print(f"[主控日志] 子查询: '{sub_query}'")
            # 使用路由器选择agent
            agent_name, confidence = router.route_query(sub_query)
            print(f"[主控日志] 匹配的agent: '{agent_name}', 匹配度: {confidence}")
            if agent_name:
                # 基于 agent_name 获取对应的 agent server
                agent_client = network.get_agent(agent_name)
                if agent_client:
                    # Message用来存储具体的任务内容
                    message = Message(content=TextContent(text=sub_query), role=MessageRole.USER)
                    # Task中存储和封装Message
                    agent_task = Task(message=message.to_dict(), id="task-" + str(uuid.uuid4()))
                    # 记录所有task：因为没有使用 await，那么这个异步函数不会被执行，会返回一个未被处理的协程对象，但里面的代码不会运行
                    tasks.append(agent_client.send_task_async(agent_task))
                    agent_names.append(agent_name)
                    confidences.append(confidence)

        confidence = sum(confidences) / len(confidences) if confidences else 0.1
        print("===========所有子查询置信度的平均值==============")
        print(confidence)

        # 3.3）并行执行任务！
        if tasks:
            # 使用 asyncio.gather 并行执行所有任务，并收集结果
            results = await asyncio.gather(*tasks, return_exceptions=True)
            print("[主控日志]检查query拆解任务之后的所有 任务结果:")
            print(results)

            # 3.4）处理和打印任务结果
            for i, result in enumerate(results):
                if isinstance(result, Exception):  # 检查结果是否是异常（任务执行失败）
                    print(f"[主控日志] {agent_names[i]} 处理错误: {str(result)}")
                else:  # 任务成功完成
                    print(f"[主控日志] {agent_names[i]} 收到完整响应：")
                    print(json.dumps(result.to_dict(), indent=4, ensure_ascii=False))  # 格式化打印完整的A2A任务响应

                    # 解析 artifacts 中的 parts
                    print(f"\n[主控日志] {agent_names[i]} 解析 artifacts 中的 parts：")
                    for artifact in result.artifacts:
                        if "parts" in artifact:  # 检查是否存在 parts 字段
                            for part in artifact["parts"]:
                                part_type = part.get("type")  # 获取 part 的类型
                                if part_type == "text":
                                    print(f"Text 结果: {part.get('text')}")
                                elif part_type == "error":
                                    print(f"Error 消息: {part.get('message')}")
                                elif part_type == "function_response":  # 如果类型是函数响应
                                    print(
                                        f"Function Response: name={part.get('name')}, response={part.get('response')}")
                                else:  # 其他未知类型
                                    print(f"未知类型: {part}")

        else:  # 如果任务列表为空
            print("[主控日志] 未找到合适代理")

        print('*'*80)
        # break


if __name__ == '__main__':
    asyncio.run(main())
```

