---
date: 2025-10-21 09:31:27
title: Coze_RAG智能体部署V2
categories: [Et_Cetera, Coze_RAG智能体部署V2]
tag: Et_Cetera
---

# 测试聊天

<div id="chat-container">
  <div id="chat-messages" style="height:300px;border:1px solid #ccc;overflow:auto;padding:10px;margin-bottom:10px;"></div>
  <input type="text" id="user-input" placeholder="输入消息..." style="width:200px;">
  <button onclick="sendMessage()">发送</button>
</div>

<script>
async function sendMessage() {
  const input = document.getElementById('user-input');
  const message = input.value;

  if (!message) return;

  addMessage('user', message);
  input.value = '请求中...';

  try {
​    const response = await fetch('https://api.coze.cn/v1/workflow/stream_run', {
​      method: 'POST',
​      headers: {
​        'Content-Type': 'application/json',
​        'Authorization': 'Bearer cztei_lPiGDcNVveCgmmDmcQ7YGf6fKWbocyCY17lYUcW0f6SKgXrgpBACqams2E7TGFC9o',
​      },
​      body: JSON.stringify({
​        workflow_id: "7537601708483346432",
​        parameters: { input: message }
​      })
​    });
​    
​    const data = await response.json();
​    console.log('API响应:', data);
​    
    input.value = '';
    addMessage('assistant', '响应: ' + JSON.stringify(data));

  } catch (error) {
​    input.value = '';
​    addMessage('assistant', '错误: ' + error.message);
  }
}

function addMessage(role, content) {
  const container = document.getElementById('chat-messages');
  const msgDiv = document.createElement('div');
  msgDiv.textContent = (role === 'user' ? '你: ' : 'AI: ') + content;
  msgDiv.style.color = role === 'user' ? 'blue' : 'green';
  msgDiv.style.margin = '5px 0';
  container.appendChild(msgDiv);
  container.scrollTop = container.scrollHeight;
}
</script>