# graphnas_dist/utils/logger.py

import os
import logging
import shutil
from typing import Dict, Any, Optional
import torch
from datetime import datetime
import pandas as pd
from pathlib import Path

class GraphNASLogger:
    """
    一个专为GraphNAS项目设计的综合日志记录器。

    功能:
    1.  **分级日志记录**: 将日志同时输出到控制台和带时间戳的文件。
    2.  **训练指标跟踪**: 自动记录和存储训练过程中的关键指标（如准确率、损失）。
    3.  **GPU内存监控**: 可选地跟踪并记录每个GPU的内存使用情况。
    4.  **结果导出**: 能够将所有记录的指标一键保存为CSV文件，便于分析和绘图。
    """
    
    def __init__(self, 
                 log_dir: str = "logs/exp",
                 name: str = "graphnas_run",
                 log_level: str = "INFO",
                 track_gpu: bool = True):
        """
        初始化日志记录器。

        Args:
            log_dir (str): 日志和结果文件的存储根目录。
            name (str): 实验名称，用于构成日志文件名的一部分。
            log_level (str): 日志级别，如 "DEBUG", "INFO", "WARNING", "ERROR"。
            track_gpu (bool): 是否启用GPU内存监控。如果CUDA不可用，此项会自动禁用。
        """
        self.log_dir = Path(log_dir)
        self.name = name
        self.metrics_history = {}
        
        # 确保日志目录存在
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置主记录器
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level.upper())
        self.logger.propagate = False # 防止日志向根记录器重复传递

        # 如果已有handlers，先移除，防止重复添加
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
            
        # 设置日志格式
        log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        date_fmt = "%Y-%m-%d %H:%M:%S"
        formatter = logging.Formatter(fmt=log_fmt, datefmt=date_fmt)
        
        # 1. 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # 2. 文件处理器 (文件名包含时间戳，避免覆盖)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = self.log_dir / f"{self.name}_{timestamp}.log"
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        self.logger.info(f"Logger '{self.name}' initialized. Logging to file: {log_file_path}")

        # GPU 监控设置
        self.track_gpu = track_gpu and torch.cuda.is_available()
        if track_gpu and not self.track_gpu:
            self.logger.warning("GPU tracking was requested, but CUDA is not available.")
        elif self.track_gpu:
            self.logger.info(f"GPU memory tracking enabled for {torch.cuda.device_count()} device(s).")
            
    def log_metrics(self, 
                    metrics: Dict[str, Any],
                    step: Optional[int] = None,
                    prefix: str = "") -> None:
        """
        记录一批训练指标，并将其存储以备后续导出。

        Args:
            metrics (Dict[str, Any]): 包含指标名称和值的字典。
            step (Optional[int]): 当前的步骤或epoch号，用于对齐数据。
            prefix (str): 添加到每个指标名称前的前缀 (例如 'train/', 'val/')。
        """
        log_parts = []
        if step is not None:
             if 'step' not in self.metrics_history:
                 self.metrics_history['step'] = []
             self.metrics_history['step'].append(step)

        for name, value in metrics.items():
            full_name = f"{prefix}{name}"
            
            # 存储历史记录
            if full_name not in self.metrics_history:
                self.metrics_history[full_name] = []
            self.metrics_history[full_name].append(value)
            
            # 准备日志消息
            if isinstance(value, float):
                log_parts.append(f"{full_name}={value:.4f}")
            else:
                log_parts.append(f"{full_name}={value}")
        
        log_message = f"Metrics (Step {step}): " + " | ".join(log_parts) if step is not None else "Metrics: " + " | ".join(log_parts)
        self.logger.info(log_message)

    def log_gpu_memory(self, step: Optional[int] = None) -> None:
        """记录当前所有可用GPU的内存使用情况。"""
        if not self.track_gpu:
            return
            
        gpu_metrics = {}
        for i in range(torch.cuda.device_count()):
            # 单位转换为 MB
            mem_alloc = torch.cuda.memory_allocated(i) / (1024 ** 2)
            mem_reserved = torch.cuda.memory_reserved(i) / (1024 ** 2)
            gpu_metrics[f'gpu{i}_alloc_mb'] = mem_alloc
            gpu_metrics[f'gpu{i}_reserved_mb'] = mem_reserved
        
        self.log_metrics(gpu_metrics, step=step, prefix="system/")

    def save_metrics_to_csv(self, filename: Optional[str] = None) -> None:
        """将所有记录的指标历史保存为 CSV 文件。"""
        if not self.metrics_history:
            self.logger.warning("No metrics have been recorded, skipping CSV export.")
            return
        
        # 确保所有列表长度一致，不一致的用None填充
        max_len = max(len(v) for v in self.metrics_history.values())
        for k in self.metrics_history:
            self.metrics_history[k].extend([None] * (max_len - len(self.metrics_history[k])))

        try:
            df = pd.DataFrame(self.metrics_history)
            
            if filename is None:
                filename = f"{self.name}_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
            csv_path = self.log_dir / filename
            df.to_csv(csv_path, index=False)
            self.logger.info(f"Metrics successfully saved to {csv_path}")
        except Exception as e:
            self.logger.error(f"Failed to save metrics to CSV: {e}")

    def close(self) -> None:
        """安全地关闭所有日志文件处理器。"""
        self.logger.info("Closing logger.")
        handlers = self.logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.logger.removeHandler(handler)

# ====================
#  独立的测试模块
# ====================
if __name__ == '__main__':
    print("===== Running Test for utils/logger.py =====")
    
    # --- 测试设置 ---
    TEST_LOG_DIR = "temp_test_logs"
    if os.path.exists(TEST_LOG_DIR):
        shutil.rmtree(TEST_LOG_DIR) # 清理旧的测试日志

    # 1. 初始化Logger
    print("\n[Test 1] Initializing logger...")
    logger = GraphNASLogger(log_dir=TEST_LOG_DIR, name="test_run", log_level="INFO")
    
    # 2. 测试基本日志记录
    print("\n[Test 2] Testing basic log messages...")
    logger.logger.debug("This is a debug message. Should not be visible with INFO level.")
    logger.logger.info("This is an info message.")
    logger.logger.warning("This is a warning message.")
    logger.logger.error("This is an error message.")

    # 3. 模拟训练循环并记录指标
    print("\n[Test 3] Simulating training loop and logging metrics...")
    for epoch in range(1, 4):
        # 模拟训练指标
        train_metrics = {'loss': 1.5 - epoch * 0.4, 'acc': 0.75 + epoch * 0.05}
        logger.log_metrics(train_metrics, step=epoch, prefix='train/')

        # 模拟验证指标
        val_metrics = {'loss': 1.3 - epoch * 0.35, 'acc': 0.78 + epoch * 0.04}
        logger.log_metrics(val_metrics, step=epoch, prefix='val/')

        # 记录GPU内存
        logger.log_gpu_memory(step=epoch)

    # 4. 保存指标到CSV
    print("\n[Test 4] Saving metrics to CSV file...")
    logger.save_metrics_to_csv()

    # 5. 关闭Logger
    logger.close()
    
    print(f"\n===== Test Finished. Check the folder '{TEST_LOG_DIR}' for logs and csv file. =====")