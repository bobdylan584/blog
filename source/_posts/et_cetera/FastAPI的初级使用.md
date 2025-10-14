---
date: 2025-01-10 04:12:45
title: FastAPI的初级使用
categories: [et_cetera, FastAPI的初级使用]
tag: et_cetera
---

# 简单的FastAPI实现

FastAPI的get方法的简单使用

注意：localhost:port/docs可以看到该ip+端口号，可以访问到的接口数量和状态

```
# FastAPI_Simple_apply.py
from fastapi import FastAPI,WebSocket

# 创建 FastAPI 应用实例，设置标题和描述
app = FastAPI(title="问答系统API", description="集成MySQL和RAG的智能问答系统")

@app.get("/greet")  # 访问该root函数内容的地址为：http://localhost:11557/greet。
async def root():
    # 在网页打印return的信息内容
    return {"message": "Hello World"}

if __name__ == "__main__":
    import uvicorn
    '''
    参数解释：
    review：本py程序文件的名字
    app：FastAPI的实例化对象名
    localhost：本机名
    port=0；代表自动选择端口号
    reload:用于开发时的自动重载功能;
    reload=True：保存文件后，服务器自动重启，立即生效
    reload=False：需要手动停止服务器，然后重新启动
    unicorn：是用来启动FastAPI的；
    '''
    uvicorn.run("FastAPI_Simple_apply:app", host="localhost", port=0, reload=False)
    # 最终效果：web端，首行打印内容：{"message": "Hello World"}
```

