#!/bin/bash

echo "===== 生成 Hexo ====="
hexo clean
hexo generate

echo "===== 打包 ====="
tar -czf site.tar.gz public

echo "===== 上传 ====="
scp site.tar.gz root@47.115.72.68:/home/projects/

echo "===== 远程解压 ====="
ssh root@47.115.72.68 "
rm -rf /home/projects/blog &&
mkdir -p /home/projects/blog &&
tar -xzf /home/projects/site.tar.gz -C /home/projects/blog --strip-components=1 &&
rm -f /home/projects/site.tar.gz
"

echo "===== 完成 ====="
