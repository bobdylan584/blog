---
date: 2025-05-21 18:32:14
title: 借助MCP搭建AI数据分析智能体
categories: [AI, 借助MCP搭建AI数据分析智能体]
tag: AI
---

# 借助MCP搭建AI数据分析智能体

进行MCP智能体快速发开发，来搭建一个能够进行SQL查询和Python自动编写的入门级数据分析智能体。

## 1、创建项目

使用Pycharm创建一个项目

## 2、配置MySQL

安装好mysql后，创建表并插入数据

```sql
CREATE DATABASE school;
USE school;
-- 然后创建一个虚拟表格，里面包含了10位同学各自3门课程的分数：
CREATE TABLE students_scores (
    id INT AUTO_INCREMENT PRIMARY KEY,  
    name VARCHAR(50),                   
    course1 INT,                        
    course2 INT,                       
    course3 INT                        
);
INSERT INTO students_scores (name, course1, course2, course3)
VALUES
    ('学生1', 85, 92, 78),
    ('学生2', 76, 88, 91),
    ('学生3', 90, 85, 80),
    ('学生4', 65, 70, 72),
    ('学生5', 82, 89, 95),
    ('学生6', 91, 93, 87),
    ('学生7', 77, 78, 85),
    ('学生8', 88, 92, 91),
    ('学生9', 84, 76, 80),
    ('学生10', 89, 90, 92);
```

## 3、创建SQL_server.py

- 功能示意图

![image-20250909185824488](https://bob-blog-image.oss-cn-shanghai.aliyuncs.com/MCP_imgs/image-20250909185824488.png)

- 代码如下：

```python
import json
import httpx
from typing import Any
import pymysql
import csv
from mcp.server.fastmcp import FastMCP

# 初始化 MCP 服务器
mcp = FastMCP("SQLServer")
USER_AGENT = "SQLserver-app/1.0"


@mcp.tool()
async def sql_inter(sql_query):
    """
    查询本地MySQL数据库，通过运行一段SQL代码来进行数据库查询。\
    :param sql_query: 字符串形式的SQL查询语句，用于执行对MySQL中school数据库中各张表进行查询，并获得各表中的各类相关信息
    :return：sql_query在MySQL中的运行结果。
    """

    connection = pymysql.connect(
        host='localhost',  # 数据库地址
        user='root',  # 数据库用户名
        passwd='root',  # 数据库密码
        db='school',  # 数据库名
        charset='utf8'  # 字符集选择utf8
    )

    try:
        with connection.cursor() as cursor:
            # SQL查询语句
            sql = sql_query
            cursor.execute(sql)

            # 获取查询结果
            results = cursor.fetchall()

    finally:
        connection.close()

    return json.dumps(results)


@mcp.tool()
async def export_table_to_csv(table_name, output_file):
    """
    将 MySQL 数据库中的某个表导出为 CSV 文件。

    :param table_name: 需要导出的表名
    :param output_file: 输出的 CSV 文件路径
    """
    # 连接 MySQL 数据库
    connection = pymysql.connect(
        host='localhost',  # 数据库地址
        user='root',  # 数据库用户名
        passwd='root',  # 数据库密码
        db='school',  # 数据库名
        charset='utf8'  # 字符集
    )

    try:
        with connection.cursor() as cursor:
            # 查询数据表的所有数据
            query = f"SELECT * FROM {table_name};"
            cursor.execute(query)

            # 获取所有列名
            column_names = [desc[0] for desc in cursor.description]

            # 获取查询结果
            rows = cursor.fetchall()

            # 将数据写入 CSV 文件
            with open(output_file, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)

                # 写入表头
                writer.writerow(column_names)

                # 写入数据
                writer.writerows(rows)

            print(f"数据表 {table_name} 已成功导出至 {output_file}")

    except Exception as e:
        print(f"导出失败: {e}")

    finally:
        connection.close()


if __name__ == "__main__":
    # 以标准 I/O 方式运行 MCP 服务器
    mcp.run(transport='stdio')
```



## 4、创建Python_server.py

功能示意图：

![image-20250909190123864](https://bob-blog-image.oss-cn-shanghai.aliyuncs.com/MCP_imgs/image-20250909190123864.png)

代码如下：

```python
import json
from typing import Any
import csv
import numpy as np
import pandas as pd
import random
from mcp.server.fastmcp import FastMCP

# 初始化 MCP 服务器
mcp = FastMCP("PythonServer")
USER_AGENT = "Pythonserver-app/1.0"


@mcp.tool()
async def python_inter(py_code):
    """
    运行用户提供的 Python 代码，并返回执行结果。

    :param py_code: 字符串形式的 Python 代码
    :return: 代码运行的最终结果
    """
    # 获取全局作用域的变量字典，后续执行 eval 或 exec 时会用到。
    g = globals()

    try:
        # 若是表达式【有返回值的语句，例如 "1+2", "len([1,2,3])"】，直接运行并返回
        result = eval(py_code, g)
        return json.dumps(str(result), ensure_ascii=False)

    except Exception:
        global_vars_before = set(g.keys())  # 记录执行前的全局变量集合
        try:
            # exec 执行语句（如 for 循环、函数定义、变量赋值等）
            exec(py_code, g)
        except Exception as e:
            return json.dumps(f"代码执行时报错: {e}", ensure_ascii=False)

        global_vars_after = set(g.keys())
        # 获取执行后的全局变量集合，找出新增加的变量
        new_vars = global_vars_after - global_vars_before

        if new_vars:
            # 只返回可序列化的变量值
            safe_result = {}
            for var in new_vars:
                try:
                    json.dumps(g[var])  # 尝试序列化，确保可以转换为 JSON
                    safe_result[var] = g[var]
                except (TypeError, OverflowError):
                    safe_result[var] = str(g[var])  # 如果不能序列化，则转换为字符串

            return json.dumps(safe_result, ensure_ascii=False)

        else:
            return json.dumps("已经顺利执行代码", ensure_ascii=False)


if __name__ == "__main__":
    # 以标准 I/O 方式运行 MCP 服务器
    mcp.run(transport='stdio')

```

## 5、创建其他服务器

- weather_server.py

```python
import json
import httpx
from typing import Any
from mcp.server.fastmcp import FastMCP

# 初始化 MCP 服务器
mcp = FastMCP("WeatherServer")

# OpenWeather API 配置
OPENWEATHER_API_BASE = "https://api.openweathermap.org/data/2.5/weather"
API_KEY = "47436a80e35d8a0f3ca9fb9cb4f2f1bc"  # 请替换为你自己的 OpenWeather API Key
USER_AGENT = "weather-app/1.0"

async def fetch_weather(city: str) -> dict[str, Any] | None:
    """
    从 OpenWeather API 获取天气信息。
    :param city: 城市名称（需使用英文，如 Beijing）
    :return: 天气数据字典；若出错返回包含 error 信息的字典
    """
    params = {
        "q": city,
        "appid": API_KEY,
        "units": "metric",
        "lang": "zh_cn"
    }
    headers = {"User-Agent": USER_AGENT}

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(OPENWEATHER_API_BASE, params=params, headers=headers, timeout=30.0)
            response.raise_for_status()
            return response.json()  # 返回字典类型
        except httpx.HTTPStatusError as e:
            return {"error": f"HTTP 错误: {e.response.status_code}"}
        except Exception as e:
            return {"error": f"请求失败: {str(e)}"}

def format_weather(data: dict[str, Any] | str) -> str:
    """
    将天气数据格式化为易读文本。
    :param data: 天气数据（可以是字典或 JSON 字符串）
    :return: 格式化后的天气信息字符串
    """
    # 如果传入的是字符串，则先转换为字典
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except Exception as e:
            return f"无法解析天气数据: {e}"

    # 如果数据中包含错误信息，直接返回错误提示
    if "error" in data:
        return f"⚠️ {data['error']}"

    # 提取数据时做容错处理
    city = data.get("name", "未知")
    country = data.get("sys", {}).get("country", "未知")
    temp = data.get("main", {}).get("temp", "N/A")
    humidity = data.get("main", {}).get("humidity", "N/A")
    wind_speed = data.get("wind", {}).get("speed", "N/A")
    # weather 可能为空列表，因此用 [0] 前先提供默认字典
    weather_list = data.get("weather", [{}])
    description = weather_list[0].get("description", "未知")

    return (
        f"🌍 {city}, {country}\n"
        f"🌡 温度: {temp}°C\n"
        f"💧 湿度: {humidity}%\n"
        f"🌬 风速: {wind_speed} m/s\n"
        f"🌤 天气: {description}\n"
    )

@mcp.tool()
async def query_weather(city: str) -> str:
    """
    输入指定城市的英文名称，返回今日天气查询结果。
    :param city: 城市名称（需使用英文）
    :return: 格式化后的天气信息
    """
    data = await fetch_weather(city)
    return format_weather(data)

if __name__ == "__main__":
    # 以标准 I/O 方式运行 MCP 服务器
    mcp.run(transport='stdio')
```

- write_server.py

```python
from mcp.server.fastmcp import FastMCP

# 初始化 MCP 服务器
mcp = FastMCP("WriteServer")



@mcp.tool()
async def write_file(content):
    """
    将指定内容写入本地文件。
    :param content: 必要参数，字符串类型，用于表示需要写入文档的具体内容。
    :return：是否成功写入
    """
    
    return "已成功写入本地文件。"


if __name__ == "__main__":
    # 以标准 I/O 方式运行 MCP 服务器
    mcp.run(transport='stdio')
```



## 6、创建MCP客户端Client

接下来考虑创建客户端Client，此时客户端需要满足以下几点要求：

- 同时连接多个服务器上的若干个工具；

- 需要能够同时完成串联或者并联模式；

- 需要能够支持多轮对话。

![ss](https://bob-blog-image.oss-cn-shanghai.aliyuncs.com/MCP_imgs/ss-17574159613601.png)

代码如下：

```python
import asyncio
import os
import json
from typing import Optional, Dict, Any, List, Tuple
from contextlib import AsyncExitStack

from openai import OpenAI
from dotenv import load_dotenv

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()


class MultiServerMCPClient:
    def __init__(self):
        """管理多个 MCP 服务器的客户端"""
        self.exit_stack = AsyncExitStack()
        self.openai_api_key = os.getenv("api_key")
        self.base_url = os.getenv("base_url")
        self.model = os.getenv("model_name")
        if not self.openai_api_key:
            raise ValueError("❌ 未找到 openai_api_key，请在 .env 文件中配置")

        # OpenAI Client（同步 API）
        self.client = OpenAI(api_key=self.openai_api_key, base_url=self.base_url)

        # 存储 (server_name -> MCP ClientSession) 映射
        self.sessions: Dict[str, ClientSession] = {}
        # 存储工具信息
        self.tools_by_session: Dict[str, list] = {}  # 每个 session 的 tools 列表
        self.all_tools: List[dict] = []  # 合并所有工具的列表

    async def connect_to_servers(self, servers: dict):
        """
        servers: {"weather": "weather_server.py", "rag": "rag_server.py"}
        """
        for server_name, script_path in servers.items():
            session = await self._start_one_server(script_path)
            self.sessions[server_name] = session

            resp = await session.list_tools()
            # resp.tools 通常是一个工具描述列表
            self.tools_by_session[server_name] = resp.tools

            for tool in resp.tools:
                # 把 MCP tool 转成 LLM 可见的工具对象（这里采用 type/function 包装）
                function_name = f"{server_name}_{tool.name}"
                # tool 里可能有: name, description, inputSchema 或 input_schema
                input_schema = getattr(tool, "inputSchema", None) or getattr(tool, "input_schema", None) or {}
                self.all_tools.append({
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "description": tool.description if hasattr(tool, "description") else "",
                        "input_schema": input_schema
                    }
                })

        # 将 input_schema 转成 OpenAI/工具系统期望的 parameters 结构
        self.all_tools = await self.transform_json(self.all_tools)

        print("\n✅ 已连接到下列服务器:")
        for name in servers:
            print(f"  - {name}: {servers[name]}")
        print("\n汇总的工具:")
        for t in self.all_tools:
            # 这里 t 的结构保留 {"type":"function","function":{...}}
            print(f"  - {t['function']['name']}")

    async def transform_json(self, json2_data):
        """
        将 {"type":"function","function":{"name":..,"description":..,"input_schema":{...}}}
        转为保持同样外层但把 input_schema -> parameters
        """
        result = []
        for item in json2_data:
            if not isinstance(item, dict) or "type" not in item or "function" not in item:
                continue
            old_func = item["function"]
            if not isinstance(old_func, dict) or "name" not in old_func or "description" not in old_func:
                continue

            new_func = {
                "name": old_func["name"],
                "description": old_func["description"],
                "parameters": {}
            }

            if "input_schema" in old_func and isinstance(old_func["input_schema"], dict):
                old_schema = old_func["input_schema"]
                new_func["parameters"]["type"] = old_schema.get("type", "object")
                new_func["parameters"]["properties"] = old_schema.get("properties", {})
                new_func["parameters"]["required"] = old_schema.get("required", [])

            result.append({
                "type": item["type"],
                "function": new_func
            })
        return result

    async def _start_one_server(self, script_path: str) -> ClientSession:
        is_python = script_path.endswith(".py")
        is_js = script_path.endswith(".js")
        if not (is_python or is_js):
            raise ValueError("服务器脚本必须是 .py 或 .js 文件")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[script_path],
            env=None
        )
        # 使用 AsyncExitStack 管理生命周期
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        read_stream, write_stream = stdio_transport
        session = await self.exit_stack.enter_async_context(ClientSession(read_stream, write_stream))
        await session.initialize()
        return session

    # helper: 把 CallToolResult 统一转换成字符串（优先结构化 data）
    def extract_tool_result(self, call_tool_result: Any) -> str:
        # 1) 优先取结构化数据（如果有）
        data = getattr(call_tool_result, "data", None)
        if data is not None:
            try:
                return json.dumps(data, ensure_ascii=False)
            except Exception:
                return str(data)

        # 2) 回退到 content blocks（常见为列表，每个 block 可能有 .text 或 .content 字段）
        content = getattr(call_tool_result, "content", None)
        if content:
            parts: List[str] = []
            for block in content:
                # block 可能是对象或 dict
                text = None
                if isinstance(block, dict):
                    text = block.get("text") or block.get("content") or str(block)
                else:
                    # 对象可能有 text 属性
                    text = getattr(block, "text", None) or getattr(block, "content", None)
                if text is None:
                    # fallback: 尝试直接 str()
                    text = str(block)
                parts.append(text)
            return "\n".join(parts)

        # 3) 兜底
        return "工具执行无输出"

    # 把 OpenAI API 的同步调用放到线程池，避免阻塞 event loop
    async def _call_openai_sync(self, /, *args, **kwargs):
        return await asyncio.to_thread(self.client.chat.completions.create, *args, **kwargs)

    async def chat_base(self, messages: list) -> Any:
        """
        messages: list of dicts: {"role":..., "content":...}
        返回 OpenAI response 对象（同步 response，但我们在线程里调用）
        """
        response = await self._call_openai_sync(
            model=self.model,
            messages=messages,
            tools=self.all_tools
        )

        # 若模型选择了调用工具( finish_reason == "tool_calls" )
        # 反复处理 tool call loop（直到模型不再要求工具）
        while getattr(response.choices[0], "finish_reason", None) == "tool_calls":
            messages = await self.create_function_response_messages(messages, response)
            response = await self._call_openai_sync(
                model=self.model,
                messages=messages,
                tools=self.all_tools
            )

        return response

    async def create_function_response_messages(self, messages, response):
        """
        解析 response 中的 tool_calls，调用 MCP 工具，并把工具结果加入消息序列
        """
        # 把 assistant 的原始消息也加入（模型的 tool_call 消息）
        # response.choices[0].message 可能为 pydantic 模型，转换为 dict
        messages.append(response.choices[0].message.model_dump() if hasattr(response.choices[0].message, "model_dump") else response.choices[0].message)

        # 遍历所有 tool_calls
        function_call_messages = response.choices[0].message.tool_calls or []
        for function_call_message in function_call_messages:
            tool_name = function_call_message.function.name
            try:
                tool_args = json.loads(function_call_message.function.arguments)
            except Exception:
                # arguments 可能已经是 dict
                tool_args = getattr(function_call_message.function, "arguments", {}) or {}

            # 运行 MCP 工具
            function_response_raw = await self._call_mcp_tool(tool_name, tool_args)
            # function_response_raw 已经是字符串（_call_mcp_tool 返回 str）
            messages.append(
                {
                    "role": "tool",
                    "content": function_response_raw,
                    "tool_call_id": function_call_message.id,
                }
            )
        return messages

    async def process_query(self, user_query: str, messages: Optional[List[Dict[str, Any]]] = None) -> Tuple[str, List[Dict[str, Any]]]:
        """
        处理单个用户查询并返回模型最终文本输出与更新后的消息历史。
        - 将 user_query append 到 messages（若 messages 为 None 则新建）。
        - 使用 chat_base() 来运行模型（chat_base 已处理 tool_calls 的循环）。
        - 返回 (final_text, messages)
        备注：返回 messages 以便外部（如 chat_loop）维护对话历史。
        """
        if messages is None:
            messages = []

        # Append user message
        messages.append({"role": "user", "content": user_query})
        # 保持历史长度（这里保留最近 20 条）
        messages = messages[-20:]

        # 使用 chat_base（会自动处理工具调用循环）
        response = await self.chat_base(messages)

        # response 为 OpenAI 的返回对象；取最终 assistant 消息文本
        assistant_msg = response.choices[0].message
        # 尝试直接取 assistant_msg.content
        final_text = getattr(assistant_msg, "content", None)
        '''
        在某些 SDK（或 Pydantic）实现里，message 可能是一个复杂的 Pydantic 对象或自定义对象，直接访问 .content 不一定存在或不可靠。很多这类对象都提供 model_dump()（或类似方法）把内部数据转换成 Python 原生 dict。
        所以当直接取不到 content 时，试着把对象“摊平”成 dict，再从 dict 里取 content
        '''
        if final_text is None and hasattr(assistant_msg, "model_dump"):
            md = assistant_msg.model_dump()
            if isinstance(md, dict):
                final_text = md.get("content")
            else:
                final_text = str(md)

        # 若仍然没有文本（理论不常见），回退为整个 message 的 str 表示
        if final_text is None:
            try:
                final_text = str(assistant_msg)
            except Exception:
                final_text = ""

        # 把 assistant 最终消息加入到历史（维持与 chat_base/工具调用一致的历史）
        # 注意：chat_base 在返回之前并没有把最后的 assistant message append 到 messages（除非在 tool loop 中）。
        # 为保证历史一致，这里做一次安全追加（若最后一条不是 assistant，则追加）
        if not messages or messages[-1].get("role") != "assistant":
            # 尝试把 pydantic message 转为普通 dict（兼容性）
            assistant_entry = assistant_msg.model_dump() if hasattr(assistant_msg, "model_dump") else assistant_msg
            messages.append(assistant_entry)

        # 截断历史，避免无限增长
        messages = messages[-40:]

        return final_text, messages

    async def _call_mcp_tool(self, tool_full_name: str, tool_args: dict) -> str:
        parts = tool_full_name.split("_", 1)
        if len(parts) != 2:
            return f"无效的工具名称: {tool_full_name}"

        server_name, tool_name = parts
        session = self.sessions.get(server_name)
        if not session:
            return f"找不到服务器: {server_name}"

        # 执行 MCP 工具（可能返回 CallToolResult）
        resp = await session.call_tool(tool_name, tool_args)
        # 把 resp 统一提取成字符串
        out = self.extract_tool_result(resp)
        return out

    async def chat_loop(self):
        print("\n🤖 多服务器 MCP + Function Calling 客户端已启动！输入 'quit' 退出。")
        # messages 作为会话历史在 chat_loop 与 process_query 之间传递
        messages: List[Dict[str, Any]] = []

        while True:
            # 使用线程调用 input，以免阻塞 event loop
            query = await asyncio.to_thread(input, "\n你: ")
            if query is None:
                continue
            query = query.strip()
            if query.lower() == "quit":
                break

            try:
                # 使用 process_query 处理本次输入（process_query 返回 final_text 与更新后的 messages）
                final_text, messages = await self.process_query(query, messages)
                # 打印模型输出
                print(f"\nAI: {final_text}")
            except Exception as e:
                print(f"\n⚠️ 调用过程出错: {e}")

    async def cleanup(self):
        await self.exit_stack.aclose()


async def main():
    servers = {
        "write": "write_server.py",
        "weather": "weather_server.py",
        "SQLServer": "SQL_server.py",
        "PythonServer": "Python_server.py"
    }

    client = MultiServerMCPClient()
    try:
        await client.connect_to_servers(servers)
        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())

```

## 7、功能测试

- 启动：

python client.py

- 测试天气：

请问今天北京的天气如何？

- 测试数据库：

```tex
请帮我查询数据库中总共包含几张表？
这张表中总共有几条数据？
请帮我将这张表导出到本地
```

- 测试python

你好，请帮我编写并运行一段Python代码，来创建一个10位的随机数

- NL2SQL+NL2Python功能联动测试

请帮我运行Python代码来读取本地students_scores.csv文件，并打印第一行数据