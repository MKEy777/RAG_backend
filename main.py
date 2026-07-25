"""
PyRAG 后端服务入口
可直接在 PyCharm 中运行此文件启动服务
"""
import os
import subprocess
import sys
import atexit
import socket

import uvicorn


def _kill_port(port: int):
    """强杀占用指定端口的进程（Windows）"""
    result = subprocess.run(["netstat", "-ano"], capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if f":{port}" in line and "LISTENING" in line:
            pid = line.split()[-1]
            if pid != "0":
                subprocess.run(["taskkill", "/PID", pid, "/F"], capture_output=True)
                print(f"  [✓] 已终止占用端口 {port} 的进程 (PID: {pid})")
                return


def _is_redis_running(host="localhost", port=6379) -> bool:
    """检测 Redis 是否运行"""
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except (ConnectionRefusedError, socket.timeout, OSError):
        return False


def _start_celery_worker():
    worker = subprocess.Popen(
        [sys.executable, "-m", "celery", "-A", "pyrag.tasks.celery_app",
         "worker", "--loglevel=info", "--pool=solo"],
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )

    def _cleanup():
        try:
            worker.terminate()
            worker.wait(timeout=5)
        except Exception:
            pass

    atexit.register(_cleanup)
    os.environ["PYRAG_CELERY_WORKER_PID"] = str(worker.pid)
    print("  [✓] Celery Worker 已启动")


if __name__ == "__main__":
    _kill_port(8000)

    print("=" * 50)
    print("  PyRAG 后端服务启动中...")
    print("=" * 50)
    print()
    print("  后端地址: http://localhost:8000")
    print("  API 文档: http://localhost:8000/docs")
    print()
    print("  前端启动命令:")
    print("    cd RAG_frontend; npm run dev")
    print()
    print("  前端地址: http://localhost:3000")
    print("=" * 50)
    print()

    # 检测 Redis 并决定是否启动 Celery
    if _is_redis_running():
        if "PYRAG_CELERY_WORKER_PID" not in os.environ:
            _start_celery_worker()
    else:
        print("  [!] Redis 未运行，跳过 Celery Worker")
        print("      文档解析任务将无法异步执行")
        print("      启动 Redis 后重启可启用完整功能")
        print()

    uvicorn.run(
        "pyrag.api.app:app",
        host="127.0.0.1",
        port=8000,
        log_level="info",
        access_log=False,
    )
