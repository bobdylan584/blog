---
title: Lunix_Commands
date: 2025-08-20 10:51:07
tag: Command_Record
---

### 

# 命令集

## Ubuntu系统

### Ubuntu 防火墙 ufw 开启并阻止了 22

```
之前运行过 ufw 相关命令，如果后来有人启用了 ufw，SSH 会被挡掉。
sudo ufw status （看状态）
Status: active 
sudo ufw allow 22 （允入22）
sudo ufw reload （重新加载）
sudo ufw disable （关闭）
```

### npm install（node packages manages）

### nano（linux文本编辑器，比vim简单）

```
nano xxx（以普通用户权限编辑文本文件）
sudo nano（以管理员权限编辑文本文件）
ctrl + K (删掉整行)
ctrl + x（退出）
enter （回车确认保存）
```

### Node.js

Node（全名 Node.js）就是让 JavaScript 能够在服务器上运行的环境。

```
http://47.115.xx.xx:3000/（跑后端 API 服务，背后就是 Node 服务器在运行。）
npm run dev:web 【跑前端编译工具（Vite、Webpack、Vue、React 都依赖 Node）】
npm install xxx （用 npm 安装库，类似pip install xxx）
node -v （判断你的系统有没有 Node）
npm -v（判断你的系统有没有 npm）
```

Node 和 JavaScript 的关系是什么？

```
JavaScript	语言（语法）
Node.js	让 JavaScript 可以跑在服务器上的工具

类比：
Python 是语言
Python 运行环境让你可以执行 .py 文件
python 语言=Python 解释器（python环境）

同理：
JS 是语言
Node 是让 JS 能跑 .js 的运行环境
JS语言 = Node解释器（node环境）
```

### 配置前端API绑定

```
VITE_API=http://40.100.70.60:3000/ pm2 start "npm run dev:web" --name "music-web"
```

### pm2（PM = Process Manager（进程管理器）版本2）

pm2 是一个让 Node 项目在服务器“长期运行、不掉、可重启”的守护进程管理工具。让 node / npm 程序后台运行（不用一直开终端），程序崩溃自动重启

```
安装pm2工具：
sudo npm install pm2@latest -g 

在home目录下创建musicuser的类似用户账号目录，并赋予pm2读写权限：
sudo mkdir -p /home/musicuser/.pm2
sudo chown -R musicuser:musicuser /home/musicuser/.pm2
sudo chmod -R 755 /home/musicuser/.pm2

查看版本号，顺便查看pm2是否读写权限授予成功：
pm2 -v 


授予musicuser用户root权限，加入sudo组，让musicuser具备使用sudo执行任何命令的权限：
sudo usermod -aG sudo musicuser

pm2的log管理工具pm2-logrotate，防止日志内存爆炸
pm2 install pm2-logrotate （一键启用自动清理pm2 的logs）
pm2 set pm2-logrotate:rotateInterval '0 0 * * *'  （每天重置日志）
pm2 set pm2-logrotate:max_size 10M （限制单个日志最大文件大小（例如 10M））
pm2 set pm2-logrotate:retain 7 （保留最近 7 个日志文件（自动删除旧的））
pm2 set pm2-logrotate:compress true （是否启用压缩（建议开启））
pm2 flush （立即清空所有日志？）

pm2 list（查看当前状态）
pm2 stop name （停止指定进程）
pm2 restart name （重新启动某进程）
pm2 delete name （用程序名删除）
pm2 stop 7 （用id停止）
pm2 delete music-web （用id删除）
pm2 save（保存启动的进程）
pm2 logs（看进程执行的日志）
pm2 start "npm run dev:web" --name "music-web"（程序管理工具打开网页开发模式，取别名）
pm2 start app.js
pm2 start "npm run dev"

启用了 PM2 的开机自启功能：
pm2 startup（服务器重启后，pm2工具会自启，里面的running的程序也会随之自启）
pm2 save


npm run start（Node 后端项目的启动命令）
pm2 logs music-api（查看该进程有没有起来）
我的项目是 Electron + 前端 + 后端 混合仓库
cat package.json（查看script字段，找到后台 API 的真正启动命令）

```

### MobaXterm快捷键

```
settings->terminal->terminal-features: √paste using right-click（启用右键点击粘贴）
按住鼠标左键选中一段命令，就是复制
```

### linux系统删除占用端口的进程

```
# -i:端口号 筛选端口，-P 显示端口数字，-t 只输出 PID
sudo lsof -i:8080 -P -t
# -9 是强制终止信号，慎用（可能导致进程数据丢失）
sudo kill -9 5678
```

### `cp` 常见用法

```
cp 是 Linux / macOS / Unix 系统中的“复制文件”命令，全称 copy。
它的作用就是：把一个文件复制成另一个文件。
e.g.	cp .env.example .env
意思是：把项目提供的 .env.example 模板文件复制一份，命名为 .env。
这是很多 Docker / Node / Python 项目常见的做法，因为：
.env.example 是示例环境变量文件
.env 是真正运行时读取的环境变量文件
为什么 .env 要复制？
开发者不会把 .env 发到 GitHub，因为里面会有私密信息。
所以会提供一个 .env.example 让你：照着写
修改自己的参数
用于本地或服务器部署

复制文件：
cp source.txt target.txt

复制文件到目录：
cp file.txt /some/path/

递归复制文件夹（必须带 -r）：
cp -r src_folder/ new_folder/
```

### docker命令

```
docker ps 查看挂起的容器的id等信息
docker stop ID
docker rm ID
docker rm -f ID 强制删除
docker logs ID 确认容器内部程序监听几号端口
docker build -t myfreebili .   把本地项目包装成镜像
docker run -d -p 8000:8000 myfreebili  运行自己的镜像
docker images （docker image ls）查看本地所有 Docker 镜像
docker rmi myfreebili （删除前必须确保所有使用该镜像的容器都已停止并删除）
docker ps -a 查看所有容器（包括停止的容器）

每次你修改本地项目，都必须重新执行以下两步
docker build -t freebili:latest .
docker stop freebili
docker rm freebili
docker run -d -p 8000:8000 freebili:latest

docker exec -it e0d1f7e3 bash  进入容器内部（最常用）
Get-Process -Name python | Stop-Process -Force 强制停止所有 Python 进程

uv sync  # 本项目使用uv管理python依赖
uv run fastapi dev main.py# 启动项目
```

### tar

```
tar -czvf 文件名.tar.gz 要压缩的文件夹 
tar -czvf backup.tar.gz file1 file2 file3   把多个文件打包成 tar.gz
tar -xzvf 文件名.tar.gz  解压缩命令
```

### scp

```
上传单个文件： scp 本地文件路径 用户名@服务器IP:/服务器路径/
scp project.tar.gz musicuser@47.115.72.68:/home/projects/

从本地上传整个文件夹到服务器：scp -r 本地文件夹路径 用户名@服务器IP:/服务器路径/
scp -r MusicPlayer/ musicuser@47.115.72.68:/home/projects/
```

### sudo

```
授予musicuser用户root权限，加入sudo组，让musicuser具备使用sudo执行任何命令的权限：
sudo usermod -aG sudo musicuser
```

### langchain降版本

```
检查当前执行的python环境是哪一个（检查路径是不是和interpreter一致）：
python -c "import sys; print(sys.executable)"

# 1. 确认 pip 属于 venv
python -m pip -V
# 2. 清干净 LangChain 新体系
python -m pip uninstall langchain langchain-core langchain-community langchain-openai -y
# 3. 装你要的老版本（支持 initialize_agent）
python -m pip install langchain==0.1.16

步骤 1：卸载当前高版本 LangChain
pip uninstall langchain -y
步骤 2：安装保留 initialize_agent 的旧版本
pip install langchain==0.3.26
步骤 3：验证版本（确保降级成功）
import langchain
print(langchain.__version__)  # 输出 0.0.350 即成功


```

