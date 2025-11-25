#!/usr/bin/env python3
import requests
import time
import json


class VLLMTester:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.model = "qwen3_moe"

    def test_health(self):
        """测试服务器健康状态"""
        try:
            response = requests.get(f"{self.base_url}/health")
            if response.status_code == 200:
                print("✅ 服务器健康状态: 正常")
                return True
            else:
                print("❌ 服务器健康状态异常")
                return False
        except Exception as e:
            print(f"❌ 无法连接到服务器: {e}")
            return False

    def test_models(self):
        """测试模型列表接口"""
        try:
            response = requests.get(f"{self.base_url}/v1/models")
            models = response.json()
            print("✅ 模型列表:", json.dumps(models, indent=2))
            return True
        except Exception as e:
            print(f"❌ 获取模型列表失败: {e}")
            return False

    def test_chat_completion(self, prompt, max_tokens=100):
        """测试聊天补全"""
        url = f"{self.base_url}/v1/chat/completions"

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.7
        }

        try:
            start_time = time.time()
            response = requests.post(url, json=payload)
            end_time = time.time()

            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                usage = result['usage']

                print(f"✅ 请求成功 (耗时: {end_time - start_time:.2f}s)")
                print(f"📝 问题: {prompt}")
                print(f"🤖 回答: {content}")
                print(f"📊 使用统计: {usage}")
                return True
            else:
                print(f"❌ 请求失败: {response.status_code}")
                print(response.text)
                return False

        except Exception as e:
            print(f"❌ 请求异常: {e}")
            return False

    def run_comprehensive_test(self):
        """运行全面测试"""
        print("=== vLLM服务器全面测试 ===\n")

        # 测试1: 健康检查
        print("1. 健康检查...")
        if not self.test_health():
            return False

        # 测试2: 模型列表
        print("\n2. 模型列表检查...")
        if not self.test_models():
            return False

        # 测试3: 多个测试问题
        test_prompts = [
            "请用中文简单介绍自己",
            "什么是人工智能？",
            "写一个Python函数计算阶乘",
            "解释一下机器学习的基本概念"
        ]

        print("\n3. 聊天补全测试...")
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n--- 测试 {i}/4 ---")
            if not self.test_chat_completion(prompt):
                return False
            time.sleep(1)  # 避免请求过快

        print("\n🎉 所有测试通过！vLLM服务器运行正常")
        return True


if __name__ == "__main__":
    tester = VLLMTester()
    tester.run_comprehensive_test()