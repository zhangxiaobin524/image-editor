#!/bin/bash
# 服务端部署脚本 - 在服务器上运行
set -e

cd "$(dirname "$0")"

echo "=== 1/4 拉取最新代码 ==="
git pull origin main

echo "=== 2/4 检查环境配置 ==="
if [ ! -f .env ]; then
    echo "❌ 缺少 .env 文件，请先创建！"
    exit 1
fi
echo "✅ .env 已就绪"

echo "=== 3/4 构建并启动 ==="
docker compose up -d --build

echo "=== 4/4 验证服务 ==="
sleep 3
if curl -sf http://localhost:9004/api/health > /dev/null 2>&1; then
    echo "✅ 图片编辑器运行正常"
else
    echo "❌ 服务异常，查看日志："
    docker compose logs --tail 20
    exit 1
fi

echo ""
docker ps --filter name=image-editor --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
echo ""
echo "✅ 部署完成！访问: http://120.55.250.184:9004"
