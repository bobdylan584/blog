---
date: 2025-10-21 09:31:27
title: Coze_RAG智能体部署
categories: [Et_Cetera, Coze_RAG智能体部署]
tag: Et_Cetera
---



# AI聊天助手

<div id="chat-container">
  <div id="chat-messages"></div>
  <input type="text" id="user-input" placeholder="输入你的问题...">
  <button onclick="sendMessage()">发送</button>
</div>

<script>
// 暴力方案 - 直接写死在页面里
async function sendMessage() {
  const input = document.getElementById('user-input');
  const message = input.value;

  if (!message) return;

  // 显示用户消息
  addMessage('user', message);
  input.value = '';

  try {
​    // 调用Coze工作流 - 替换成你的真实信息！
​    const response = await fetch('https://api.coze.cn/v1/workflow/stream_run', {
​      method: 'POST',
​      headers: {
​        'Content-Type': 'application/json',
​        'Authorization': 'Bearer cztei_qpOoUeGOPWhvk1tXCbmnANsUXbYPLsa9SpaKy3AYLAgXoSi17oTntCWJWOI5VWw20',
​      },
​      body: JSON.stringify({
​        workflow_id: "7537601708483346432",
​        parameters: {
​          "input": "你好，请帮我推荐旅游景点"
​        }
​      })
​    });
​    

    const data = await response.json();
    // 显示AI回复
    addMessage('assistant', data.messages[0].content);

  } catch (error) {
​    addMessage('assistant', '出错了：' + error.message);
  }
}

function addMessage(role, content) {
  const container = document.getElementById('chat-messages');
  const msgDiv = document.createElement('div');
  msgDiv.className = role;
  msgDiv.textContent = content;
  container.appendChild(msgDiv);
}
</script>

<style>
#chat-container {
  border: 1px solid #ccc;
  padding: 20px;
  margin: 20px 0;
}
#chat-messages {
  height: 300px;
  overflow-y: auto;
  border: 1px solid #eee;
  margin-bottom: 10px;
  padding: 10px;
}
.user { color: blue; }
.assistant { color: green; }
</style>