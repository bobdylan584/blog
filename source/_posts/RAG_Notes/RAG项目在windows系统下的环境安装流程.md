---
date: 2025-03-07 11:21:51
title: RAG项目在windows系统下的环境安装流程
categories: [RAG_Notes, RAG项目在windows系统下的环境安装流程]
tag: RAG_Notes
---

# RAG项目环境安装流程

## 1 python虚拟环境

- 依赖文件 (`requirements.txt`)

    > 自己创建requirements.txt文件, 将以下内容复制进去即可

    ```properties
    # Web框架
    fastapi
    uvicorn[standard]
    websockets
    
    # 数据库连接
    pymysql
    redis
    
    # 向量检索
    rank_bm25
    scikit-learn
    numpy
    jieba
    pandas
    transformers
    torch
    langchain
    langchain_community
    sentence_transformers
    pymilvus
    
    # AI模型
    openai
    
    # 配置和工具
    configparser
    locust
    websocket-client
    
    # 其他依赖
    pydantic
    starlette
    ```

- 打开cmd终端(win+r)输入以下命令

    ```sh
    # 创建虚拟环境, 安装3.10及以上的python解析器
    conda create -n EduRAG python=3.10
    # 切换虚拟环境 
    conda activate EduRAG
    # 安装依赖包, 在requirements.txt对应路径下执行以下命令
    pip install -r requirements.txt 
    ```

    ![1753622116646](https://bob-blog-image.oss-cn-shanghai.aliyuncs.com/RAG/rag_env_settings/1753622116646.png)

## 2 安装Docker Desktop

- 安装WSL（Windows Subsystem for Linux）

    - **启用WSL**打开具有管理员权限的PowerShell，运行以下命令以安装WSL

        ```sh
        wsl --install
        ```

        ![1753623042372](https://bob-blog-image.oss-cn-shanghai.aliyuncs.com/RAG/rag_env_settings/1753623042372.png)

        ![1753623117795](https://bob-blog-image.oss-cn-shanghai.aliyuncs.com/RAG/rag_env_settings/1753623117795.png)

- 下载Docker Desktop

    - 访问[Docker官方网站](https://www.docker.com/products/docker-desktop/)，下载适用于Windows/Mac的Docker Desktop安装包。
- 安装Docker Desktop
    - 双击下载的安装文件，按照提示一路下一步完成安装
    - 安装完成后，重启计算机

- 验证Docker安装

    - 打开PowerShell或命令提示符，运行：

        ```
        docker --version
        docker compose version
        ```

        ![1753623823676](https://bob-blog-image.oss-cn-shanghai.aliyuncs.com/RAG/rag_env_settings/1753623823676.png)

## 3 配置Docker Compose文件

- 创建项目目录

    - 在本地磁盘（例如:C盘）创建一个文件夹，用于存放Milvus和Redis的配置文件和数据：	

        ![1753623917487](https://bob-blog-image.oss-cn-shanghai.aliyuncs.com/RAG/rag_env_settings/1753623917487.png)

    - 创建Docker Compose文件, 添加以下内容 (先创建.txt格式文件, 然后修改为.yml格式)

        ![1753623986186](https://bob-blog-image.oss-cn-shanghai.aliyuncs.com/RAG/rag_env_settings/1753623986186.png)

        ```properties
        # version: '3.5'
        
        # Windows/x86_64 Version
        # Redis Password set to: 1234
        
        services:
          etcd:
            container_name: milvus-etcd
            image: quay.io/coreos/etcd:v3.5.5
            environment:
              - ETCD_AUTO_COMPACTION_MODE=revision
              - ETCD_AUTO_COMPACTION_RETENTION=1000
              - ETCD_QUOTA_BACKEND_BYTES=4294967296
              - ETCD_ENABLE_V2=true
            volumes:
              - ./volumes/etcd:/etcd/data
            command: etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd/data
        
          minio:
            container_name: milvus-minio
            image: minio/minio:RELEASE.2023-03-20T20-16-18Z
            environment:
              MINIO_ACCESS_KEY: minioadmin
              MINIO_SECRET_KEY: minioadmin
            volumes:
              - ./volumes/minio:/minio/data
            command: minio server /minio/data
            healthcheck:
              test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
              interval: 30s
              timeout: 20s
              retries: 3
        
          standalone:
            container_name: milvus-standalone
            # 使用多架构镜像，Docker on Windows 会自动拉取 x86_64 版本
            image: milvusdb/milvus:v2.4.4
            command: ["milvus", "run", "standalone"]
            environment:
              - ETCD_ENDPOINTS=etcd:2379
              - MINIO_ADDRESS=minio:9000
            volumes:
              - ./volumes/milvus:/var/lib/milvus
            ports:
              - "19530:19530"
              - "9091:9091"
            depends_on:
              - "etcd"
              - "minio"
        
          redis:
            container_name: milvus-redis
            image: redis:latest
            restart: always
            ports:
              - "6379:6379"
            volumes:
              - ./volumes/redis:/data
            # 设置 Redis 密码为 1234
            command: redis-server --requirepass 1234
        
        networks:
          default:
            name: milvus-network
        ```

## 4 启动Milvus和Redis

- 拉取镜像并启动容器

    - 在milvus_redis目录下，打开具有管理员权限的PowerShell终端，运行：

        ```sh
        # -d 表示后台运行容器
        docker compose up -d
        ```

        ![1753625803111](https://bob-blog-image.oss-cn-shanghai.aliyuncs.com/RAG/rag_env_settings/1753625803111.png)

        ![1753625673169](https://bob-blog-image.oss-cn-shanghai.aliyuncs.com/RAG/rag_env_settings/1753625673169.png)

    - 拉取镜像时报错: Get "https://registry-1.docker.io/v2/": net/http: request canceled while waiting for connection (Client.Timeout exceeded while awaiting headers)

        CSDN链接:https://blog.csdn.net/jhgj56/article/details/142209517?fromshare=blogdetail&sharetype=blogdetail&sharerId=142209517&sharerefer=PC&sharesource=weixin_45056110&sharefrom=from_link

        ```json
        "registry-mirrors": [
            "https://docker.211678.top",
            "https://docker.1panel.live",
            "https://hub.rat.dev",
            "https://docker.m.daocloud.io",
            "https://do.nark.eu.org",
            "https://dockerpull.com",
            "https://dockerproxy.cn",
            "https://docker.awsl9527.cn"
        ]
        ```

        ![1753625438227](https://bob-blog-image.oss-cn-shanghai.aliyuncs.com/RAG/rag_env_settings/1753625438227.png)

        > 重启docker生效

## 5 验证Milvus和Redis

- 测试Milvus连接

    - 创建Python脚本（例如 test_milvus.py ）

        ```python
        from pymilvus import connections, utility
        
        # 连接到 Milvus
        connections.connect(host="localhost", port="19530")
        
        # 检查版本
        print(f"Milvus version: {utility.get_server_version()}")
        ```

    - 在RAG虚拟环境下运行脚本

        ```sh
        # cmd终端执行命令
        # 输出类似Milvus version: 2.4.10 表示Milvus部署成功
        python test_milvus.py
        ```

- 测试Redis连接

    - 创建Python脚本（例如 test_redis.py ）

        ```python
        import redis
        
        # 连接到 Redis
        client = redis.Redis(host="localhost", port=6379, password=1234, decode_responses=True)
        
        # 测试读写
        client.set("test_key", "Hello, Redis!")
        value = client.get("test_key")
        print(f"Redis value: {value}")
        ```

    - 在RAG虚拟环境下运行脚本

        ```sh
        # cmd终端执行命令
        # 输出类似Redis value: Hello, Redis! 表示Redis部署成功
        python test_redis.py
        ```

## Readme

```
问题一、安装requirement内的三方包失败：
报错信息：
WARNING: Connection timed out while downloading.
ERROR: Could not install packages due to an OSError: [WinError 32] 另一个程序正在使用此文件，进程无法访问。: 'C:\\Users\\username\\AppData\\Local\\Temp\\pip-unpack-ivzdh5h9\\transformers-4.54.0-py3-none-any.whl'
Consider using the `--user` option or check the permissions.

前置条件：
在管理员权限的 CMD 下运行命令

大模型解决方案：
case1：解决连接超时问题和解决文件占用问题：
pip install --user -r D:\tool\ITCast\EduRAGEnvInstallDescribe\requirements.txt -i https://mirrors.aliyun.com/pypi/simple/


问题二、安装wsl --install失败（Windows Subsystem for Linux）
现象：返回：被禁止
解决：
启用相关Windows功能：按Win+R键，输入“optionalfeatures.exe”并回车，打开“启用或关闭Windows功能”窗口。确保勾选“适用于Linux的Windows子系统”和“虚拟机平台”选项，然后点击“确定”进行安装，安装完成后重启计算机。
网络要稳定，换手机移动网。

问题三、配置Docker Compose文件失败
现象1：获取镜像文件失败
解决1：网络不稳定
解决2：打开docker desktop；进入设置-docker engine-添加
"registry-mirrors": [
    "https://docker.211678.top",
    "https://docker.1panel.live",
    "https://hub.rat.dev",
    "https://docker.m.daocloud.io",
    "https://do.nark.eu.org",
    "https://dockerpull.com",
    "https://dockerproxy.cn",
    "https://docker.awsl9527.cn"
]

问题四、执行docker compose up -d提示端口被占用
报错信息：
Error response from daemon: ports are not available: exposing port TCP 0.0.0.0:9091 -> 127.0.0.1:0: listen tcp 0.0.0.0:9091: bind: Only one usage of each socket address (protocol/network address/port) is normally permitted.

netstat -ano | findstr :9091
taskkill /PID 32360 /F
```

