---
date: 2025-05-05 12:34:22
title: MCP协议
categories: [AI_Module, MCP协议]
tag: AI_Module
---

**MCP协议**

MCP模型上下文协议

**学习目标**

- 理解协议在AI中的作用。
- 掌握MCP的核心概念及应用。
- 分析MCP与Function Calling的差异，并解释为什么MCP可以降低对大模型Function Calling能力的要求。

**1、什么是MCP协议**

MCP（Model Context Protocol，模型上下文协议）是由 Anthropic 在2024年1月提出的一套开放协议，旨在实现大型语言模型（LLM）与外部数据源和工具的无缝集成，用来在大模型和数据源之间建立安全双向的链接。

Anthropic的愿景，希望把MCP协议打造成AI世界的“Type-C”接口，可以通过MCP协议工具、数据链接起来，达类似HTTP协议的那种通用程度。

**1.1核心概念**

**协议的定义：**

协议（Protocol）是一种约定或标准，用于定义不同系统、设备或软件之间如何通信和交换数据。它确保各方使用相同的“语言”和规则，避免混乱。例如，HTTP协议定义了浏览器与服务器的交互方式，USB协议标准化了设备连接。

**协议在AI中的作用**：

在大型语言模型（LLM）和AI代理系统中，协议用于标准化模型与外部工具、数据源或其它代理的交互。AI系统越来越复杂，需要模型访问外部资源（如数据库、API）或与其他AI协作，而协议就是“桥梁”，确保高效、安全的通信。

以下是**MCP vs. 传统API (Flask/REST)**：

![img](https://bob-blog-image.oss-cn-shanghai.aliyuncs.com/MCP_imgs/image1.png)

**1.2 MCP角色**

MCP协议有两个核心角色：客户端与服务端。

**MCP服务端 (Tool Provider)**：

**角色**：工具的提供者。

**职责**：将一个或多个本地函数（例如，Python函数）包装起来，通过一个标准的MCP接口暴露出去。它监听来自客户端的请求，执行对应的函数，并返回结果。

**例子**：一个天气查询服务、一个数学计算服务、一个数据库访问服务。

**MCP客户端 (Tool Consumer)**：

**角色**：工具的调用者或消费者。

**职责**：连接到MCP服务端，查询可用的工具列表（自发现），并根据需要调用这些工具。

**例子**：大模型Agent、自动化脚本、任何需要远程执行功能的应用程序。

![img](https://bob-blog-image.oss-cn-shanghai.aliyuncs.com/MCP_imgs/image2.png)

**1.3 MCP的通信传输方式**

MCP协议本身与传输方式无关，mcp及python-a2a库等提供了两种常见的实现：stdio和基于HTTP的SSE。

**stdio (标准输入/输出)**：

stdio一种非常经典和简单的进程间通信（IPC）方式。客户端启动服务端作为一个**子进程**。

客户端通过写入子进程的**标准输入 (stdin)** 来发送请求。

客户端通过读取子进程的**标准输出 (stdout)** 来接收响应。

日志和错误信息通常输出到**标准错误 (stderr)**，以避免干扰主通信。

**SSE (Server-Sent Events)**

sses是一种基于**HTTP**的协议，允许服务端向客户端单向推送事件（消息）。

客户端向服务器发起一个普通的HTTP请求。

服务器保持这个连接打开，并以一种特殊的文本格式 (text/event-stream) 不断地向客户端发送数据。

它比WebSocket更轻量，非常适合服务器向客户端单向推送更新的场景。在MCP中，它被用来实现请求-响应模式。

**如何选择：stdio还是SSE？**

![img](https://bob-blog-image.oss-cn-shanghai.aliyuncs.com/MCP_imgs/image3.png)

**1.4 MCP与大模型Function Calling**

**Function Calling**（或称Tool Calling）是**大模型自身具备的一项核心能力**。它指的是模型在处理用户输入时，能够：

1. **理解意图**：识别出用户的请求需要借助外部工具或函数才能完成。
2. **选择工具**：从提供给它的工具列表中，选择最合适的一个或多个。
3. **提取参数**：从用户的自然语言中，准确地抽取出调用该工具所需的参数。
4. **生成结构化输出**：以一种严格的、可供程序解析的格式（通常是JSON）返回它决定调用的函数名和参数。

**这个能力是模型通过在海量代码和文本上进行预训练而获得的。** 模型学会了将自然语言指令映射到结构化的函数调用上。

![img](https://bob-blog-image.oss-cn-shanghai.aliyuncs.com/MCP_imgs/image4.png)

什么情况下，必须使用MCP Server?

1、工具太多了，需要进行规范和统一。

2、需要使用别人开发好的工具/工具包，来完成AI应用的快速搭建！

**2、代码示例**

**MCP服务端**

mcp服务端角色是工具

**职责**：将一个或多个本地函数（例如，Python函数）包装起来，通过一个标准的MCP接口暴露出去。它监听来自客户端的请求，执行对应的函数，并返回结果。

```python3
Python
# courseware/mcp_demo/mcp_server.py
import uvicorn
from python_a2a.mcp import FastMCP, create_fastapi_app

# 1. 创建一个 FastMCP 实例
mcp = FastMCP(
    name="GreeterServer",
    description="一个使用 FastMCP 实现的问候工具服务。",
    version="1.0.0"
)

# 2. 使用 @mcp.tool() 装饰器，将一个函数注册为“工具”
@mcp.tool(
    name="greet",
    description="根据姓名生成一句问候语。"
)
def greet(name: str) -> str:
    """这个函数现在是一个标准化的工具了。"""
    print(f"[服务器日志] 'greet' 工具被调用，参数 name = {name}")
    response = f"你好, {name}! 欢迎来到 FastMCP 的世界。"
    print(f"[服务器日志] 执行完成，返回结果：'{response}'")
    return response
```

**MCP客户端**

mcp客户端的角色是工具的调用者或消费者。

**职责**：连接到MCP服务端，查询可用的工具列表（自发现），并根据需要调用这些工具。

```python3
Python
# courseware/mcp_demo/mcp_client.py
import asyncio
from python_a2a.mcp import MCPClient

async def main():
    server_url = "http://127.0.0.1:6001"
    print(f"[客户端日志] 准备连接到 MCP 服务器: {server_url}")
    client = MCPClient(server_url)

    try:
        # (自发现) 获取服务器上的工具列表
        tools = await client.get_tools()
        print(f"[客户端日志] 通过'自发现'得知服务器有以下工具:")
        for tool in tools:
            print(f"  - {tool.get('name')}: {tool.get('description')}")

        # 调用工具
        tool_name_to_call = "greet"
        tool_params = {"name": "王五"}
        print(f"\n[客户端日志] 准备调用工具 '{tool_name_to_call}'，参数为 {tool_params}")

        result = await client.call_tool(tool_name_to_call, **tool_params)

        print(f"[客户端日志] 收到服务器的响应: '{result}'")

    finally:
        # 关闭客户端会话
        await client.close()
```

**3、总结**

**本小结主要介绍了MCP协议的概念以及代码实现。**