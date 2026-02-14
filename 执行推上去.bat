@echo off
set GIT_BASH="D:\software\tools\Git\bin\bash.exe"
%GIT_BASH% --login -i "%~dp0deploy.sh"
pause
