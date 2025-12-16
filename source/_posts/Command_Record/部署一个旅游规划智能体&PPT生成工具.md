---
date: 2025-08-27
title: 部署一个旅游规划智能体
categories: [Command_Record, 部署一个旅游规划智能体]
tag: Command_Record
---

# 部署一个旅游规划智能体

# 在宿主机 build，Docker 只负责“装盒子”

```
核心思想一句话：
👉 所有吃内存的事情（uv / pip / 编译）都在宿主机做
👉 Docker 里只 COPY 成果，不编译、不解析、不拉依赖

1、在宿主机准备好 Python + uv
python3 --version
结果：Python 3.13.9

2、在 Windows 上安装 uv（一次性）
方法一（官方推荐，最稳）
1）打开 PowerShell（不是 CMD）：
irm https://astral.sh/uv/install.ps1 | iex
2）安装完成后，重开一个终端，验证：
uv --version
3）在 SmartVoyage 项目中初始化 uv 环境
cd \SmartVoyage
让 uv 接管这个项目
uv init
会生成一个最基础的：
pyproject.toml
4）在项目根目录创建虚拟环境
uv venv
5）用 requirements.txt 安装依赖（关键）
uv pip install -r requirements.txt
这一步会发生三件事：在项目目录下创建虚拟环境、安装所有依赖、不污染系统Python
6）确认 uv 虚拟环境已经就绪
uv run python --version
如果能正常输出 Python 版本，说明 uv 环境已经生效。

3、运行app.py主程序
uv run python app.py
如果你的 app.py 是 Streamlit 页面（大概率是：uv run streamlit run app.py
3.1）如何确认当前真的在用 uv 的 Python？
uv run python -c "import sys; print(sys.executable)"
看到类似：
D:nCode\SmartVoyage\.venv\Scripts\python.exe
而 不是：
C:\Python313\python.exe




```

解决uv报错问题

```
1、执行uv pip install -r requirements.txt，失败，没有venv环境
报错：error: No virtual environment found;
解决：
uv venv（在项目根目录创建虚拟环境）
uv pip install -r requirements.txt

2、执行uv pip install streamlit失败，
报错：Failed to extract archive: pyarrow-22.0.0-cp313-cp313-win_amd64.whl
解决：pyarrow-22.0.0-cp313-cp313-win_amd64.whl（cp313 = Python 3.13）
解决：降版本
1）确认你现在的 Python 版本
Python 3.13.x
2）用 uv 创建指定 Python 版本的虚拟环境
uv venv --python 3.10

3、执行uv venv --python 3.10失败，（项目配置文件明确“禁止”使用 Python 3.10）
报错：which is incompatible with the project's Python requirement: `>=3.13`
解决：改配置文件pyproject.toml，requires-python = ">=3.13"
为：requires-python = ">=3.10,<3.12"
3.1）执行uv venv --python 3.10失败，（uv 仍然被一个“钉死的 Python 3.13 配置文件.python-version”强制拉回 3.13）
error: The Python request from '.python-version'
解决：用 uv 正式“改钉子”
uv python pin 3.10
rmdir /s /q .venv
uv venv
uv run python --version





```

# PPT生成工具

```
前置：Python（建议 3.12）、Node.js（建议 18/20）、Docker Desktop
1) Python 依赖（项目根目录）
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .\libs\ppt_manager


) Python 依赖（项目根目录）
python -m venv .venv.\.venv\Scripts\activatepip install -r requirements.txtpip install -e .\libs\ppt_manager
如果你是 git 子模块方式引入 libs/ppt_manager，先：
git submodule update --init
2) Windows 必装的外部工具（不装会在模板分析/多模态验证时报错）
LibreOffice：安装后确保能找到 soffice.exe
代码会自动查找默认路径（例如 C:\Program Files\LibreOffice\program\soffice.exe）。
Poppler（给 pdf2image 用）：推荐最省事用 conda：
  conda install -c conda-forge poppler
或手动下载 Poppler for Windows 并把 bin 加到 PATH。
3) 复制环境文件
把 .env.example 复制成 .env，至少填 OPENAI_API_KEY。（也可以先不填，后面用管理后台改“模型配置”。）
4) 启 Redis + MLflow（只起这俩）
docker compose -f docker/docker-compose-dev.yml up -d
5) 起后端 FastAPI（项目根目录）
uvicorn apps.api.main:app --reload --host 0.0.0.0 --port 8000
健康检查：http://localhost:8000/api/v1/health
API 文档：http://localhost:8000/api/docs
6) 起 Celery（项目根目录；Windows 必须用 --pool=solo）
celery -A apps.api.celery_app worker --loglevel=info --concurrency=2 --pool=solo -Q celery,template_analysis,ppt_generation
7) 起前端（新开一个终端）
cd apps\webnpm install --legacy-peer-depsnpm run dev
访问：http://localhost:5173
（前端已在 vite.config.ts 里把 /api、/workspace 代理到 http://localhost:8000，一般不会有跨域问题）
8) 登录与最小闭环
管理后台默认账号：admin / admin123
先去管理后台配置 LLM/Vision/DeepThink 的 API Key
上传 PPT 模板 → 等分析完成 → 输入 Markdown → 生成 PPT
你现在卡在哪一步？
把你执行的命令 + 报错贴出来（尤其是 uvicorn / celery / poppler / soffice 相关），我可以按你的报错逐条带你排。
```

