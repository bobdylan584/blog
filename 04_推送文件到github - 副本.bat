@echo off
chcp 65001 > nul
echo ========================================
echo          Git 自动化提交工具
echo ========================================
echo.

echo 第一步：添加所有更改到暂存区，执行代码：git add .
git add .
if %errorlevel% neq 0 (
    echo 错误：添加文件失败！
    pause
    exit /b 1
)
echo 添加完成！
echo.

:input_commit
setlocal
set /p commit_msg="第二步：请输入本次修改的详细描述："
if "%commit_msg%"=="" (
    echo 提交信息不能为空，请重新输入！
    echo.
    endlocal
    goto input_commit
)

echo.
echo 第三步：正在提交更改...;执行如下代码
git commit -m "%commit_msg%"
if %errorlevel% neq 0 (
    echo 错误：提交失败！可能没有需要提交的更改。
    endlocal
    pause
    exit /b 1
)
endlocal
echo 提交成功！
echo.

echo 第四步：正在推送到远程仓库...
git push origin master:main
if %errorlevel% neq 0 (
    echo 错误：推送失败！请检查网络或权限。
    pause
    exit /b 1
)
echo 推送成功！
echo.

echo 所有操作已完成！
pause