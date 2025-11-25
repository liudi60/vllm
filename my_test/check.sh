#!/bin/bash
echo "=== vLLM服务器测试 ==="

# 测试健康状态
echo "1. 健康检查..."
curl -s http://localhost:8000/health && echo " OK." || echo " FAIL!"

# 测试聊天接口
echo "2. 聊天接口测试..."
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3_moe",
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 50
  }' | python3 -m json.tool

echo "测试完成！"


#给执行权限并运行：
#chmod +x test_vllm.sh
#./test_vllm.sh
