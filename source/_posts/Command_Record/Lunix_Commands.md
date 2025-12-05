---
title: Lunix_Commands
date: 2025-08-20 10:51:07
tag: Lunix_Commands
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

pm2 startup （把pm2进程管理工具，放到服务器的开机自启名单里）

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

### 让这个MusicPlayer音乐播放器在浏览器里访问

```
生成了可部署的网页版本
1、进入系统根目录
cd /home/xxx/MusicPlayer
2、安装依赖（跳过如果已经安装过）
npm install
3、生产环境构建（非常关键！）
npm run build
 3.1、Error: EACCES: permission denied 没有写入权限
 sudo chown -R musicuser:musicuser /home/xxx/MusicPlayer


把网页部署到服务器，让你能在浏览器访问。
1、Electron + Vite 项目构建后，网页文件会位于：
out/renderer/

方案 A（最快速）：用 serve 在服务器直接跑网页
1）安装 serve（如果没装）
npm install -g serve
2）运行网页
serve -s out/renderer -l 8005
3）访问
http://47.115.72.68:8005

要看到“有歌曲、有数据的成功案例
方案 A — 最快成功：用项目自带本地音乐资源
通常目录是：
/src/assets/music
/src/assets/data/music.json
out/renderer/assets/...



```

### 让这个MusicPlayer音乐播放器在浏览器里访问

```
const server = require('netease-cloud-music-api-alger/server');
const cors = require('cors');
const express = require('express');

const app = express();
app.use(cors());

const port = process.env.PORT || 3000;

server.serveNcmApi({
  app,
  port
}).then(() => {
  console.log(`后端 API 服务器运行在 http://localhost:${port}`);
}).catch(error => {
  console.error('启动后端失败:', error);
});

```



```
1、在 Ubuntu 创建 start-api.js
cd /home/xxx/MusicPlayer
nano start-api.js
插入上述文件内容

2、安装后端依赖
npm install netease-cloud-music-api-alger
npm install express cors

3、运行后端
pm2 start start-api.js --name music-api
pm2 save
确认 3000 端口已打开：
sudo ss -tulpn | grep 3000
应该看到：
node ... *:3000

4、修改前端 VITE API 地址
nano src/renderer/.env.production
写入：
VITE_API=http://47.115.72.68:3000
VITE_API_MUSIC=http://47.115.72.68:3000
然后重建前端：
npm run build

5、最终
1）安装 serve（如果没装）
npm install -g serve
2）运行网页
serve -s out/renderer -l 8005
3）访问
http://47.115.72.68:8005
4)用 PM2 运行 serve(开机自启、远程工具关闭照常运行)
cd /home/xxx/MusicPlayer
pm2 start "serve -s out/renderer -l 8005" --name music-web


```

