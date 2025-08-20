@echo off
echo ╔══════════════════════════════╗
echo ║        生成Hexo静态文件        ║
echo ╚══════════════════════════════╝

echo.
echo [1/3] 清理旧文件...
hexo clean

echo.
echo [2/3] 等待5秒...
TIMEOUT /T 5 /NOBREAK >nul

echo.
echo [3/3] 生成新文件...
hexo g

echo.
echo ✅ 生成完成！
echo 📁 文件已准备好，可以部署或启动服务器
echo.
pause