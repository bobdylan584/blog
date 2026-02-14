@echo off
echo ===== 开始生成 Hexo =====
call hexo clean
call hexo generate

echo ===== 开始打包，把public下的文件打包成site.tar.gz =====
tar -czvf site.tar.gz public

echo ===== 开始上传 =====
scp site.tar.gz root@47.115.72.68:/home/projects/

echo ===== 远程解压 =====
ssh root@47.115.72.68 "cd /home/projects && rm -rf blog && mkdir blog && tar -xzvf site.tar.gz -C blog --strip-components=1 && rm -f site.tar.gz"

echo ===== 部署完成 =====
pause
