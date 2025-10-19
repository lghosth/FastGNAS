import os
import psutil
import time
import argparse
import subprocess
from datetime import datetime
from typing import Optional, Dict, List
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
try:
    import ray
    HAS_RAY = True
except ImportError:
    HAS_RAY = False

class MemoryMonitor:

    def __init__(self, interval: float=1.0, log_file: Optional[str]=None):
        self.interval = interval
        self.start_time = time.time()
        if log_file is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            log_dir = os.path.join(script_dir, 'log')
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = os.path.join(log_dir, f'memory_{timestamp}.csv')
            print(f'📝 日志文件: {log_file}')
        self.log_file = log_file
        with open(self.log_file, 'w') as f:
            f.write('timestamp,elapsed_time,cpu_used_gb,cpu_total_gb,cpu_percent,')
            for i in range(8):
                f.write(f'gpu{i}_used_mb,gpu{i}_total_mb,gpu{i}_percent,')
            f.write('ray_object_store_gb\n')

    def get_cpu_memory(self):
        mem = psutil.virtual_memory()
        return {'used_gb': mem.used / 1000000000.0, 'total_gb': mem.total / 1000000000.0, 'percent': mem.percent, 'available_gb': mem.available / 1000000000.0}

    def get_gpu_memory(self):
        gpu_info = {}
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=2)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    parts = line.split(',')
                    if len(parts) >= 3:
                        gpu_idx = int(parts[0].strip())
                        used_mb = float(parts[1].strip())
                        total_mb = float(parts[2].strip())
                        gpu_info[f'gpu{gpu_idx}'] = {'allocated_mb': used_mb, 'total_mb': total_mb, 'percent': used_mb / total_mb * 100 if total_mb > 0 else 0}
                return gpu_info if gpu_info else None
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            pass
        if HAS_TORCH and torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_info[f'gpu{i}'] = {'allocated_mb': torch.cuda.memory_allocated(i) / 1000000.0, 'reserved_mb': torch.cuda.memory_reserved(i) / 1000000.0, 'total_mb': torch.cuda.get_device_properties(i).total_memory / 1000000.0}
            return gpu_info
        return None

    def get_ray_memory(self):
        if not HAS_RAY or not ray.is_initialized():
            return None
        try:
            resources = ray.cluster_resources()
            object_store = resources.get('object_store_memory', 0) / 1000000000.0
            return {'object_store_gb': object_store}
        except Exception:
            return None

    def print_status(self):
        elapsed = time.time() - self.start_time
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cpu_mem = self.get_cpu_memory()
        print(f"\n{'=' * 60}")
        print(f'[{timestamp}] 运行时间: {elapsed:.1f}s')
        print(f"{'=' * 60}")
        print(f"CPU内存: {cpu_mem['used_gb']:.2f}GB / {cpu_mem['total_gb']:.2f}GB ({cpu_mem['percent']:.1f}%)")
        print(f"  可用: {cpu_mem['available_gb']:.2f}GB")
        gpu_mem = self.get_gpu_memory()
        if gpu_mem:
            for gpu_id, info in gpu_mem.items():
                used_mb = info['allocated_mb']
                total_mb = info['total_mb']
                percent = info.get('percent', used_mb / total_mb * 100 if total_mb > 0 else 0)
                print(f'{gpu_id.upper()}内存: {used_mb:.1f}MB / {total_mb:.1f}MB ({percent:.1f}%)')
                if 'reserved_mb' in info:
                    print(f"  已预留: {info['reserved_mb']:.1f}MB")
        else:
            print('GPU: 不可用或nvidia-smi未找到')
        ray_mem = self.get_ray_memory()
        if ray_mem:
            print(f"Ray对象存储: {ray_mem['object_store_gb']:.2f}GB")
        else:
            print('Ray: 未初始化')
        if self.log_file:
            ray_store = ray_mem['object_store_gb'] if ray_mem else 0
            csv_line = f'{timestamp},{elapsed:.1f},'
            csv_line += f"{cpu_mem['used_gb']:.2f},{cpu_mem['total_gb']:.2f},{cpu_mem['percent']:.1f},"
            for i in range(8):
                gpu_key = f'gpu{i}'
                if gpu_mem and gpu_key in gpu_mem:
                    info = gpu_mem[gpu_key]
                    used_mb = info['allocated_mb']
                    total_mb = info['total_mb']
                    percent = info.get('percent', used_mb / total_mb * 100 if total_mb > 0 else 0)
                    csv_line += f'{used_mb:.1f},{total_mb:.1f},{percent:.1f},'
                else:
                    csv_line += '0,0,0,'
            csv_line += f'{ray_store:.2f}\n'
            with open(self.log_file, 'a') as f:
                f.write(csv_line)

    def monitor_loop(self):
        print('开始内存监控...')
        print(f'监控间隔: {self.interval}秒')
        if self.log_file:
            print(f'日志文件: {self.log_file}')
        print('按 Ctrl+C 停止监控')
        try:
            while True:
                self.print_status()
                time.sleep(self.interval)
        except KeyboardInterrupt:
            print('\n\n监控已停止')
            if self.log_file:
                print(f'日志已保存到: {self.log_file}')

def main():
    parser = argparse.ArgumentParser(description='FastGNAS 内存监控工具')
    parser.add_argument('--interval', '-i', type=float, default=1.0, help='监控间隔（秒），默认1.0')
    parser.add_argument('--log', '-l', type=str, default=None, help='日志文件路径（可选，默认为 tools/log/memory_YYYYMMDD_HHMMSS.csv）')
    args = parser.parse_args()
    monitor = MemoryMonitor(interval=args.interval, log_file=args.log)
    monitor.monitor_loop()
if __name__ == '__main__':
    main()