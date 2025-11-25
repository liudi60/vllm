#!/usr/bin/env python3
import threading
import time
import psutil
import os


def monitor_vllm_threads():
    """监控vLLM进程的线程"""
    current_pid = os.getpid()
    process = psutil.Process(current_pid)

    print(f"=== vLLM进程线程监控 (PID: {current_pid}) ===")
    print(f"进程名: {process.name()}")
    print(f"线程数: {process.num_threads()}")
    print(f"CPU使用率: {process.cpu_percent()}%")
    print(f"内存使用: {process.memory_info().rss / 1024 ** 2:.2f} MB")

    print("\n=== 活动线程 ===")
    for thread in process.threads():
        print(f"线程ID: {thread.id}, CPU时间: {thread.cpu_time}")

    # 获取Python线程信息
    print("\n=== Python线程 ===")
    for thread in threading.enumerate():
        print(f"线程: {thread.name} (ID: {thread.ident})")


if __name__ == "__main__":
    monitor_vllm_threads()