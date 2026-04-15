"""
日志工具：终端输出同时写入文件

用法：
    from utils.logger import setup_logger
    setup_logger("./outputs/exp_name/train.log")
    # 之后所有 print() 自动同时写入文件
"""
import sys
import os
import time


class DualOutput:
    """同时写入终端和文件"""
    def __init__(self, filepath, mode='a'):
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        self.terminal = sys.stdout
        self.log = open(filepath, mode, encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def setup_logger(log_path):
    """
    调用后所有 print() 同时输出到终端和 log_path。
    """
    dual = DualOutput(log_path, mode='a')
    sys.stdout = dual
    sys.stderr = DualOutput(log_path.replace('.log', '_err.log'), mode='a')

    print(f"\n{'='*60}")
    print(f"  Log started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Log file: {os.path.abspath(log_path)}")
    print(f"{'='*60}\n")

    return dual