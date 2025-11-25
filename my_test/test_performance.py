import requests
import time
import concurrent.futures


def stress_test():
    """压力测试：并发请求"""
    url = "http://localhost:8000/v1/chat/completions"

    def send_request(request_id):
        payload = {
            "model": "qwen3_moe",  # vllm启动时指定的model_id
            "messages": [{"role": "user", "content": f"这是测试请求 {request_id}"}],
            "max_tokens": 50,
            "temperature": 0.1
        }

        start_time = time.time()
        try:
            response = requests.post(url, json=payload, timeout=30)
            end_time = time.time()

            if response.status_code == 200:
                return True, end_time - start_time
            else:
                return False, 0
        except Exception as e:
            return False, 0

    # 并发测试
    print("开始压力测试（10个并发请求）...")
    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(send_request, i) for i in range(10)]     # 提交10个请求
        results = [f.result() for f in concurrent.futures.as_completed(futures)]  # as_completed 按顺序等待每个请求处理完成

    end_time = time.time()

    success_count = sum(1 for success, _ in results if success)
    total_time = end_time - start_time

    print(f"压力测试结果:")
    print(f"成功率: {success_count}/10")
    print(f"总耗时: {total_time:.2f}秒")
    print(f"平均响应时间: {total_time / 10:.2f}秒")


if __name__ == "__main__":
    stress_test()