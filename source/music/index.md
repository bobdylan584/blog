---
title: 我的音乐库
date: 2024-10-24 00:00:00
comments: false  # 可选：关闭评论
---
<head>
  <!-- APlayer 核心样式和脚本 -->
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/aplayer@1.10.1/dist/APlayer.min.css">
  <script src="https://cdn.jsdelivr.net/npm/aplayer@1.10.1/dist/APlayer.min.js"></script>
  <!-- MetingJS 扩展（简化列表配置） -->
  <script src="https://cdn.jsdelivr.net/npm/meting@2.0.1/dist/Meting.min.js"></script>
</head><!-- 音乐播放器容器 -->
<div id="music-player" style="max-width: 800px; margin: 0 auto; padding: 20px;"></div>

<script>
  // 定义本地音乐列表（路径对应 source/music_files 下的文件）
  const musicList = [
    {
      name: "歌曲1",       // 歌曲名
      artist: "歌手1",    // 歌手名
      url: "/music_files/song1.mp3",  // 音乐路径（必须以 / 开头，对应 source 目录）
      cover: "/music_files/cover1.jpg" // 可选：封面图（需放在 music_files 目录）
    },
    {
      name: "歌曲2",
      artist: "歌手2",
      url: "/music_files/song2.mp3",
      cover: "/music_files/cover2.jpg"
    }
    // 继续添加更多歌曲...
  ];

  // 初始化 APlayer 播放器
  const ap = new APlayer({
    container: document.getElementById('music-player'),
    listFolded: false,  // 默认展开播放列表
    listMaxHeight: 300, // 列表最大高度
    music: musicList    // 加载本地音乐列表
  });
</script>





