---
date: 2025-12-10
title: MCP协议被捐给了一个基金会，Anthropic是发起方
categories: [News, MCP协议被捐给了一个基金会，Anthropic是发起方]
tag: News
---

# 解读：

## function call 与 MCP 的根本区别（以及为什么 MCP 会诞生）

```
一句“总括本质”的话：
MCP 的诞生，是为了消灭各大模型之间「工具调用格式不统一」导致的碎片化开发痛点，而 function call 只是每家大模型“自己的工具调用规范”，无法跨模型复用。

1、function call（工具调用）的问题：每个模型都不一样，所以碎片化严重
大模型厂商（OpenAI、Anthropic、Google、DeepSeek、字节、月之暗面…）各自定义了自己的 function call 规范：
例如：
OpenAI 的 function calling：json_schema + name + args
Anthropic 的 tool use：tool_choice + tool_input
Google Gemini 的 tools：有自己的 schema
DeepSeek 工具调用：又是另一套格式
阿里 Qwen tools：再次不同
开发者必须针对每个模型写独立适配代码

要同时支持 OpenAI + Qwen + DeepSeek，就要做：
openai_tools_adapter.py
qwen_tools_adapter.py
deepseek_tools_adapter.py
anthropic_tools_adapter.py

而且最最重要的是，各大模型 API 经常变，做好的工具调用也会随之“全盘失效”。
这就是语料里说的：
每个大模型都有自己的工具调用格式 → 难以通用
每个数据源都要写独立连接器 → 易碎、易失效

function call 本质是“厂商私有协议”。
这就是为什么「Agent 自主处理复杂任务」在现实中很难落地。

2、MCP（Model Context Protocol）的根本意义：统一所有模型的工具调用协议
MCP 是由 Anthropic 推出的 跨大模型通用协议，但被整个行业迅速接纳（OpenAI、Llama、Cursor、Claude Desktop、VS Code 插件都支持）。

核心价值只有一句：MCP 让“工具调用”成为一个全行业统一标准。
也就是说，无论你用：OpenAI（ChatGPT）、Claude（Anthropic）或者其他模型
它们都将通过同一种方式调用工具：
tools/
   weather/
   mysql/
   filesystem/
   search/
工具 = MCP server
LLM = MCP client
两者通过统一协议通信
你写一个工具，就可以被所有支持 MCP 的 LLM 直接调用，不需要改任何代码。
这就是 MCP 能诞生的关键背景：function call 各自为政 → 太碎片 → 必须做一个统一协议（类似 Web 标准）

3、function call vs MCP 的根本区别
项目		function call			MCP
本质		厂商私有协议			行业统一协议
标准化	    不同模型格式不一样	  完全统一、跨模型复用
适配成本   每个模型都要写一次	 工具写一次，所有模型可用
稳定性		API 改一点就挂		协议稳定、工具不变
扩展性		只能做参数调用		  可建服务（DB、FS、搜索、插件…）
生态		 各搞各的			 MCP 会像 HTTP 一样成为标准
愿景		 给 LLM 用工具		 给开发者与 LLM 建立生态协议

一句话区别：
function call = 让大模型会用工具
MCP = 让所有大模型用“同一种工具协议”

4、为什么 MCP 会火？
因为以前情况如下：
❌ 一个天气 API，你要写 5 份适配（因为 5 家模型格式不同）
❌ 一个数据库连接器，你要写 5 份适配
❌ 一个本地文件系统插件，你要写 5 份适配

而到了 MCP：
✅ 工具写一次 → 所有模型通用
✅ 模型更新 → 工具完全不用改
✅ 开发者只需要学一种协议
这就像：
function call = 各国自己定电话线标准
MCP = 国际统一电话标准（任何国家的电话都能互通）

5、MCP 为什么现在才诞生？
因为 AI Agent 时代到来，问题集中爆发：
工具越来越多
多模型协作越来越常见
模型厂商 function call 越改越碎
生态严重割裂
Anthropic 抓住窗口：推出了 通用协议 MCP，并把它开源成行业基础设施。
OpenAI 很快跟进支持 MCP（极罕见的行为，意味着这是真标准）。
```

## MCP 的出现(2024年 11月 25日) ，25年3月被广泛使用，并不会让function call消失。

```
MCP 是“在 function call 之上补全整个生态的协议层”，二者是协同关系，而不是替代关系。
1. 为什么 function call 不会消失？
因为 function call 是大模型内部“如何调用一个工具”的最基础能力，它解决的问题是：
“大模型如何向开发者传递调用某个函数/工具所需要的参数？”
这是每个大模型推理架构里必不可少的一环。
模型必须有一种原生方式来描述「我要调用哪个函数、参数是什么」。
所以：
OpenAI 会继续用自己的 function calling schema
Qwen、DeepSeek、Gemini 也会继续有自己版本的 tool calling
未来所有模型都会内置某种形式的 function call
这是 模型能力层（Model Capability Layer），不会被移除。

2. 那 MCP 的角色是什么？
MCP 不是来替掉 function call 的，它解决 function call 永远无法解决的问题：
不同模型的工具调用格式不统一 → 工具无法跨平台复用 → 开发者维护成本爆炸。
所以 MCP 的定位是：
**→ function call：模型内部语言（各说各话）
→ MCP：行业通用协议（统一外部世界）**

MCP 是行业层协议（像 HTTP、USB、SQL），是“跨模型的统一标准”。

3. function call 与 MCP 的工作分层（非常重要）
第 1 层：function call（模型内部能力）
模型决定：
我要调用哪个工具
参数是什么样
工具名的格式
参数 JSON 的 schema
→ 每家模型的格式依然不一样。

第 2 层：MCP（模型与工具之间的协议层）
MCP 决定：
工具如何向模型注册（统一格式）
工具如何返回数据（统一格式）
工具如何在所有模型之间复用（统一协议）
工具如何被任何 LLM 自动发现（auto-discovery）
→ 工具只需写一次，可以给所有模型用。

```



# 正文

在刚刚结束的“美国 AI 春晚” AWS re:Invent 2025 大会上，AI Agent（智能代理）的重要性被反复提及。

亚马逊云科技 CEO Matt Garman 宣布的 12 项有关 AI 的新发布，都围绕着 Agent 的基建、开发和管理。在演讲中，他下了一个判断：AI Agent 的出现，正在让 AI 的价值真正释放。

![MCP协议被捐给了一个基金会，Anthropic是发起方](https://www.aitntnews.com/pictures/2025/12/11/078b4199-d660-11f0-a694-fa163e47d677.webp)

“让 Agent 自主处理复杂任务”——这个未来看似美好，可实现起来困难重重。因为在现实的代码世界里，开发者面临着碎片化的难题。每个大模型都有自己的工具调用格式，每个数据源都需要单独编写连接器，且随时可能因为厂商的 API 变动而失效。

不过，这一困局似乎迎来了它的转折点。

12 月 9 日，Linux 基金会牵头 OpenAI、Anthropic 等科技公司，正式宣布成立 Agentic AI Foundation（代理人工智能基金会，简称 AAIF）。

![MCP协议被捐给了一个基金会，Anthropic是发起方](https://www.aitntnews.com/pictures/2025/12/11/08648049-d660-11f0-be86-fa163e47d677.webp)

更令人意外的是，Anthropic 宣布将其重要工具—— Model Context Protocol（模型上下文协议，MCP）捐赠给该基金会。 一同被捐赠的还有OpenAI 的AGENTS.md 和 Block 的goose 框架。

# 开发者需要的不是更多模型，而是通用标准

要理解此次 Anthropic 捐赠的意义，须从开头提及的开发者面临的碎片化难题说起。

过去的一年里，AI 行业虽然在模型能力上突飞猛进，但在应用层的基础设施建设上却显得步履蹒跚。开发者在构建智能代理时，往往发现自己陷入了重复造轮子的困境。要让一个 AI 助手既能读取本地的 SQLite 数据库，又能操作 Slack 发送消息，还需要连接 Google Drive 读取文档，开发者必须针对所使用的模型（无论是 GPT-4、Claude 3.5 还是Llama）分别适配。

MCP 的出现原本就是为了解决这一痛点。一年前，Anthropic 在推出该协议时，将其定位为连接 AI 模型与数据源、工具之间的通用接口。

我们可以将 MCP 理解为计算机领域的 USB 接口：只要鼠标、键盘或打印机遵循 USB 标准，就可以插在任何品牌的电脑上使用。同样，只要数据源或工具支持 MCP 标准，任何支持该协议的 AI 模型（客户端）都能直接调用它们，而无需开发者编写复杂的中间层代码。

Anthropic 首席产品官迈克·克里格（Mike Krieger）在谈及此次捐赠时坦言，MCP 最初只是为了解决其内部团队面临的数据连接难题。但在 2024 年 11 月开源后，该协议迅速获得了社区的拥护，目前已有超过 10,000 个活跃的 MCP 服务器，涵盖了从企业级数据库到个人开发者工具的广泛场景。

![MCP协议被捐给了一个基金会，Anthropic是发起方](https://www.aitntnews.com/pictures/2025/12/11/0901cf25-d660-11f0-b1a8-fa163e47d677.webp)

此次将 MCP 移交至中立的 Linux 基金会，彻底消除了开发者的最大顾虑——供应商锁定。MCP 从此成为一个由社区共同治理的开放标准，任何厂商都无法单方面控制其发展，这为行业广泛采纳扫除了部分障碍。

# Agent 铁三角：连接、规范与运行

仅有连接标准并不够，构建可靠的 Agent 还需要行为规范与执行框架。AAIF 的另外两个创始项目——OpenAI 的 AGENTS.md 和Block 的 goose——恰好补齐了 MCP 之外的拼图，共同构成了开发者构建代理应用的三大支柱。

OpenAI 捐赠的 AGENTS.md 解决的是“上下文与行为规范”的问题。

在复杂的软件工程中，让 AI 代理接管代码库并非易事。AI 需要知道项目的架构、编码风格以及哪些文件是禁区。AGENTS.md 提供了一种极其轻量级的解决方案：开发者只需在项目根目录下放置一个标准化的 Markdown 文件，就能向任何访问该项目的 AI 代理“自报家门”。

这种方式避免了将大量上下文硬编码在提示词（Prompt）中，既节省了 Token 成本，又提高了代理执行任务的准确性。据 OpenAI 技术人员尼克·库珀（Nick Cooper）介绍，这种格式已被超过 60,000 个开源项目采用。

![MCP协议被捐给了一个基金会，Anthropic是发起方](https://www.aitntnews.com/pictures/2025/12/11/09987f34-d660-11f0-9653-fa163e47d677.webp)

而 Block 捐赠的 goose 则是一个本地优先的代理框架。

它能够直接运行在开发者的机器上，利用 MCP 连接各种工具，并遵循 AGENTS.md 的指引来执行任务。Block 的 AI 技术负责人布拉德·阿克森（Brad Axen）指出，goose 并非实验品，而是已经在 Block 内部经过数千名工程师验证的生产级工具。对于那些不想从零开始构建代理运行时的开发者来说，goose 提供了一个现成的、开源的起点。

![MCP协议被捐给了一个基金会，Anthropic是发起方](https://www.aitntnews.com/pictures/2025/12/11/0a2cc205-d660-11f0-8698-fa163e47d677.webp)

至此，技术框架已经十分明晰。开发者可以利用 goose 作为躯体，通过 MCP 作为神经系统感知外部世界，并遵循 AGENTS.md 这一大脑皮层中的规则进行决策。这种分层架构的标准化，将极大地降低 Agentic AI 的开发门槛。

# 行业巨头的集体背书

此次 AAIF 的支持阵容，除了三家创始成员外，亚马逊 AWS、谷歌、微软、Cloudflare 和彭博社等科技巨头均作为白金会员加入。这种跨越竞争关系的合作在科技史上并不多见。

Linux 基金会执行董事吉姆·赞林（Jim Zemlin）强调，AAIF 的目标是避免未来出现“封闭的围墙花园”，即工具连接、代理行为和编排都被锁定在少数几个专有平台上。对于 AWS 这样的云服务商来说，支持开放标准符合其长远利益。AWS 代理 AI 副总裁斯瓦米·西瓦苏布拉马尼亚（Swami Sivasubramanian）表示，将 MCP 等项目置于中立基金会，能让企业客户有信心在这些标准上构建关键业务流程，而不必担心底层协议的碎片化。

对于开发者来说，这意味着他们所掌握的技能将具有更强的通用性。学会编写 MCP 服务器，意味着你的工具可以同时服务于 Claude、ChatGPT、GitHub Copilot 以及未来可能出现的任何 AI 平台。这种技能的可迁移性，正如当年掌握 HTTP 协议或 SQL 语言一样，将成为 AI 时代开发者的核心竞争力。

# 从代码到生态：Agent 的未来

随着标准的统一， Agent 领域有望迎来类似移动互联网初期的应用爆发。AAIF 已经规划了后续的生态建设路径，包括定于 2026 年 4 月在纽约举行的首届官方 MCP 开发者峰会。围绕这些开放标准正在形成一个庞大的开发者社区。

OpenAI 的尼克·库珀在采访中提到，协议本质上是一种共享语言，它的价值在于沟通与协作。当数以万计的工具和服务都开始讲同一种“MCP 语言”，当所有的代码库都有了“AGENTS.md 说明书”，AI Agent 将不再是演示视频中那个笨拙的实验品，而能真正穿梭于复杂的数字系统中，成为开发者得力的助手。

参考链接：

1.https://www.anthropic.com/news/donating-the-model-context-protocol-and-establishing-of-the-agentic-ai-foundation

2.https://www.linuxfoundation.org/press/linux-foundation-announces-the-formation-of-the-agentic-ai-foundation



















