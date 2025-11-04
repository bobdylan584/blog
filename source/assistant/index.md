---
title: 阿甘小助手
date: 2025-11-04
---

{% raw %}
<div id="coze-assistant"></div>

<script>
(function(){
  var script = document.createElement("script");
  script.src = "https://lf26-cdn-tos.bytecdntp.com/obj/byte-tos/static-resource/coze-widget.js";
  script.defer = true;
  script.onload = function() {
    new CozeWebSDK({
      bot_id: "7568773837157253158",
      title: "阿甘小助手",
      theme: "light",
      bubble_text: "💬 问问阿甘？",
      bubble_icon: "🤖"
    }).init();
  };
  document.body.appendChild(script);
})();
</script>
{% endraw %}
