<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>我的音乐空间 - 酷狗风格播放器</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: "Microsoft YaHei", sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
    
        .music-player {
            max-width: 1200px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            overflow: hidden;
        }
    
        .player-header {
            background: linear-gradient(135deg, #ff6b6b, #ee5a24);
            color: white;
            padding: 30px;
            text-align: center;
        }
    
        .player-header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        }
    
        .player-header p {
            font-size: 1.1em;
            opacity: 0.9;
        }
    
        .player-body {
            display: flex;
            min-height: 500px;
        }
    
        .sidebar {
            width: 300px;
            background: #2c3e50;
            color: white;
            padding: 20px;
        }
    
        .playlist {
            list-style: none;
        }
    
        .playlist li {
            padding: 15px;
            margin: 8px 0;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
        }
    
        .playlist li:hover {
            background: rgba(255, 255, 255, 0.2);
            transform: translateX(5px);
        }
    
        .playlist li.active {
            background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        }
    
        .playlist .song-number {
            width: 25px;
            height: 25px;
            background: rgba(255, 255, 255, 0.2);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-right: 10px;
            font-size: 0.8em;
        }
    
        .main-content {
            flex: 1;
            padding: 30px;
            background: white;
        }
    
        .album-art {
            text-align: center;
            margin-bottom: 30px;
        }
    
        .album-art img {
            width: 250px;
            height: 250px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
            object-fit: cover;
        }
    
        .song-info {
            text-align: center;
            margin-bottom: 30px;
        }
    
        .song-title {
            font-size: 2em;
            color: #2c3e50;
            margin-bottom: 10px;
            font-weight: bold;
        }
    
        .song-artist {
            font-size: 1.3em;
            color: #7f8c8d;
        }
    
        .progress-area {
            margin: 30px 0;
        }
    
        .progress-bar {
            height: 6px;
            background: #ecf0f1;
            border-radius: 10px;
            cursor: pointer;
            margin-bottom: 10px;
            position: relative;
        }
    
        .progress {
            height: 100%;
            background: linear-gradient(135deg, #ff6b6b, #ee5a24);
            border-radius: 10px;
            width: 0%;
            transition: width 0.1s ease;
        }
    
        .time {
            display: flex;
            justify-content: space-between;
            color: #7f8c8d;
            font-size: 0.9em;
        }
    
        .controls {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 30px;
            margin: 30px 0;
        }
    
        .control-btn {
            width: 50px;
            height: 50px;
            border: none;
            border-radius: 50%;
            background: linear-gradient(135deg, #ff6b6b, #ee5a24);
            color: white;
            font-size: 1.2em;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
        }
    
        .control-btn:hover {
            transform: scale(1.1);
            box-shadow: 0 5px 15px rgba(255, 107, 107, 0.4);
        }
    
        .control-btn.play-pause {
            width: 70px;
            height: 70px;
            font-size: 1.5em;
        }
    
        .volume-control {
            display: flex;
            align-items: center;
            gap: 15px;
            margin-top: 20px;
        }
    
        .volume-slider {
            flex: 1;
            height: 4px;
            background: #ecf0f1;
            border-radius: 10px;
            cursor: pointer;
            position: relative;
        }
    
        .volume-progress {
            height: 100%;
            background: linear-gradient(135deg, #ff6b6b, #ee5a24);
            border-radius: 10px;
            width: 80%;
        }
    
        .lyrics-container {
            background: #f8f9fa;
            border-radius: 15px;
            padding: 20px;
            margin-top: 30px;
            max-height: 200px;
            overflow-y: auto;
        }
    
        .lyrics {
            text-align: center;
            line-height: 2;
            color: #2c3e50;
        }
    
        .lyrics .current {
            color: #ff6b6b;
            font-weight: bold;
            font-size: 1.1em;
        }
    
        /* 响应式设计 */
        @media (max-width: 768px) {
            .player-body {
                flex-direction: column;
            }
            
            .sidebar {
                width: 100%;
                order: 2;
            }
            
            .main-content {
                order: 1;
            }
            
            .album-art img {
                width: 200px;
                height: 200px;
            }
        }
    </style>
</head>
<body>
​    <div class="music-player">
​        <div class="player-header">
​            <h1>🎵 我的音乐空间</h1>
​            <p>享受音乐的每一刻</p>
​        </div>
​        
        <div class="player-body">
            <div class="sidebar">
                <h3 style="margin-bottom: 20px; color: #ff6b6b;">播放列表</h3>
                <ul class="playlist">
                    <li class="active">
                        <span class="song-number">1</span>
                        <div>
                            <div class="song-name">我要你默默走不回头</div>
                            <div class="artist-name">队长的小斑鸠、向晚晚</div>
                        </div>
                    </li>
                    <li>
                        <span class="song-number">2</span>
                        <div>
                            <div class="song-name">万千花蕊慈母悲哀</div>
                            <div class="artist-name">珂拉琪</div>
                        </div>
                    </li>
                    <li>
                        <span class="song-number">3</span>
                        <div>
                            <div class="song-name">晴天</div>
                            <div class="artist-name">周杰伦</div>
                        </div>
                    </li>
                    <li>
                        <span class="song-number">4</span>
                        <div>
                            <div class="song-name">光年之外</div>
                            <div class="artist-name">邓紫棋</div>
                        </div>
                    </li>
                </ul>
            </div>
            
            <div class="main-content">
                <div class="album-art">
                    <img src="https://bob-blog-image.oss-cn-shanghai.aliyuncs.com/audio/album-cover.jpg" alt="专辑封面" onerror="this.src='data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjUwIiBoZWlnaHQ9IjI1MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjMzQzOTRGIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCIgZm9udC1zaXplPSIyNCIgZmlsbD0iI0ZGRiIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPuWbvuWDj+Wkp+WtpjwvdGV4dD48L3N2Zz4='">
                </div>
                
                <div class="song-info">
                    <h1 class="song-title">我要你默默走不回头</h1>
                    <div class="song-artist">队长的小斑鸠、向晚晚</div>
                </div>
                
                <div class="progress-area">
                    <div class="progress-bar" id="progressBar">
                        <div class="progress" id="progress"></div>
                    </div>
                    <div class="time">
                        <span class="current-time" id="currentTime">0:00</span>
                        <span class="total-time" id="totalTime">0:00</span>
                    </div>
                </div>
                
                <div class="controls">
                    <button class="control-btn" id="prevBtn">⏮</button>
                    <button class="control-btn play-pause" id="playPauseBtn">▶</button>
                    <button class="control-btn" id="nextBtn">⏭</button>
                </div>
                
                <div class="volume-control">
                    <span>🔊</span>
                    <div class="volume-slider" id="volumeSlider">
                        <div class="volume-progress" id="volumeProgress"></div>
                    </div>
                </div>
                
                <div class="lyrics-container">
                    <div class="lyrics" id="lyrics">
                        <div class="current">我要你默默走不回头</div>
                        <div>就这样消失在街的尽头</div>
                        <div>不要回头看我泪流</div>
                        <div>就让回忆随风飘走</div>
                        <div>我要你默默走不回头</div>
                        <div>就这样消失在夜的尽头</div>
                    </div>
                </div>
                
                <!-- 隐藏的音频元素 -->
                <audio id="audioPlayer" style="display: none;">
                    <source src="https://bob-blog-image.oss-cn-shanghai.aliyuncs.com/audio/%E9%98%9F%E9%95%BF%E7%9A%84%E5%B0%8F%E6%96%91%E9%B8%A0%E3%80%81%E5%90%91%E6%99%9A%E6%99%9A%20-%20%E6%88%91%E8%A6%81%E4%BD%A0%E9%BB%98%E9%BB%98%E8%B5%B0%E4%B8%8D%E5%9B%9E%E5%A4%B4%20(%E8%AF%B4%E5%94%B1%E7%89%88).mp3" type="audio/mpeg">
                    您的浏览器不支持音频元素
                </audio>
            </div>
        </div>
    </div>
    
    <script>
        // 获取DOM元素
        const audioPlayer = document.getElementById('audioPlayer');
        const playPauseBtn = document.getElementById('playPauseBtn');
        const progressBar = document.getElementById('progressBar');
        const progress = document.getElementById('progress');
        const currentTimeEl = document.getElementById('currentTime');
        const totalTimeEl = document.getElementById('totalTime');
        const volumeSlider = document.getElementById('volumeSlider');
        const volumeProgress = document.getElementById('volumeProgress');
        const playlistItems = document.querySelectorAll('.playlist li');
        const lyrics = document.getElementById('lyrics');
    
        // 播放/暂停功能
        playPauseBtn.addEventListener('click', () => {
            if (audioPlayer.paused) {
                audioPlayer.play();
                playPauseBtn.innerHTML = '⏸';
            } else {
                audioPlayer.pause();
                playPauseBtn.innerHTML = '▶';
            }
        });
    
        // 进度条更新
        audioPlayer.addEventListener('timeupdate', () => {
            const currentTime = audioPlayer.currentTime;
            const duration = audioPlayer.duration;
            
            if (duration) {
                const progressPercent = (currentTime / duration) * 100;
                progress.style.width = `${progressPercent}%`;
                
                // 更新时间显示
                currentTimeEl.textContent = formatTime(currentTime);
                totalTimeEl.textContent = formatTime(duration);
                
                // 歌词高亮（简单模拟）
                updateLyricsHighlight(currentTime);
            }
        });
    
        // 点击进度条跳转
        progressBar.addEventListener('click', (e) => {
            const rect = progressBar.getBoundingClientRect();
            const percent = (e.clientX - rect.left) / rect.width;
            audioPlayer.currentTime = percent * audioPlayer.duration;
        });
    
        // 音量控制
        volumeSlider.addEventListener('click', (e) => {
            const rect = volumeSlider.getBoundingClientRect();
            const percent = (e.clientX - rect.left) / rect.width;
            audioPlayer.volume = percent;
            volumeProgress.style.width = `${percent * 100}%`;
        });
    
        // 播放列表点击
        playlistItems.forEach((item, index) => {
            item.addEventListener('click', () => {
                // 移除其他active类
                playlistItems.forEach(i => i.classList.remove('active'));
                // 添加当前active类
                item.classList.add('active');
                
                // 这里可以添加切换歌曲的逻辑
                // 暂时只是模拟
                audioPlayer.play();
                playPauseBtn.innerHTML = '⏸';
            });
        });
    
        // 格式化时间
        function formatTime(seconds) {
            const min = Math.floor(seconds / 60);
            const sec = Math.floor(seconds % 60);
            return `${min}:${sec.toString().padStart(2, '0')}`;
        }
    
        // 歌词高亮更新（模拟）
        function updateLyricsHighlight(currentTime) {
            const lines = lyrics.querySelectorAll('div');
            lines.forEach((line, index) => {
                line.classList.remove('current');
                // 简单模拟：每10秒切换一行
                if (Math.floor(currentTime / 10) === index) {
                    line.classList.add('current');
                    // 滚动到当前歌词
                    line.scrollIntoView({ behavior: 'smooth', block: 'center' });
                }
            });
        }
    
        // 初始化音量
        audioPlayer.volume = 0.8;
        volumeProgress.style.width = '80%';
    </script>
</body>
</html>