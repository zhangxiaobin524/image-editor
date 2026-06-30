#!/bin/bash
# 部署 image-editor 到服务器
set -e

SERVER="root@120.55.250.184"
REMOTE_DIR="/opt/fluent-life/image-editor"
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== 1/4 创建远程目录 ==="
ssh $SERVER "mkdir -p $REMOTE_DIR/server/static"

echo "=== 2/4 同步文件 ==="
rsync -avz --delete \
  --exclude '__pycache__' \
  --exclude '.git' \
  --exclude 'chroma_db' \
  "$LOCAL_DIR/docker-compose.yml" \
  "$LOCAL_DIR/.env" \
  "$LOCAL_DIR/server/" \
  $SERVER:$REMOTE_DIR/server/

echo "=== 3/4 构建启动 ==="
ssh $SERVER "cd $REMOTE_DIR && docker compose up -d --build"

echo "=== 4/4 验证 ==="
ssh $SERVER "docker ps --filter name=image-editor --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'"
echo ""
echo "✅ 部署完成！访问: http://120.55.250.184:9004"
