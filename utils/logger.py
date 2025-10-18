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
    A comprehensive logger for the project.

    Features:
    1.  Logs to both console and a timestamped file.
    2.  Tracks and stores key training metrics (e.g., accuracy, loss).
    3.  Optionally monitors and records GPU memory usage.
    4.  Exports all recorded metrics to a CSV file for analysis.
    """
    
    def __init__(self, 
                 log_dir: str = "logs/exp",
                 name: str = "graphnas_run",
                 log_level: str = "INFO",
                 track_gpu: bool = True):
        """
        Initializes the logger.

        Args:
            log_dir (str): Root directory for logs and results.
            name (str): Experiment name for the log filename.
            log_level (str): Logging level ("DEBUG", "INFO", "WARNING", "ERROR").
            track_gpu (bool): Enable GPU memory monitoring.
        """
        self.log_dir = Path(log_dir)
        self.name = name
        self.metrics_history = {}
        
        # Ensure log directory exists
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup main logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level.upper())
        self.logger.propagate = False # Prevent duplicate logs in root logger

        # Remove existing handlers to avoid duplication
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
            
        # Set log format
        log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        date_fmt = "%Y-%m-%d %H:%M:%S"
        formatter = logging.Formatter(fmt=log_fmt, datefmt=date_fmt)
        
        # 1. Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # 2. File handler (with timestamp)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = self.log_dir / f"{self.name}_{timestamp}.log"
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        self.logger.info(f"Logger '{self.name}' initialized. Logging to file: {log_file_path}")

        # GPU tracking setup
        self.track_gpu = track_gpu and torch.cuda.is_available()
        if track_gpu and not self.track_gpu:
            self.logger.warning("GPU tracking requested, but CUDA not available.")
        elif self.track_gpu:
            self.logger.info(f"GPU memory tracking enabled for {torch.cuda.device_count()} device(s).")
            
    def log_metrics(self, 
                    metrics: Dict[str, Any],
                    step: Optional[int] = None,
                    prefix: str = "") -> None:
        """
        Records a batch of metrics.

        Args:
            metrics (Dict[str, Any]): Dictionary of metric names and values.
            step (Optional[int]): Current step or epoch number.
            prefix (str): Prefix for metric names (e.g., 'train/', 'val/').
        """
        log_parts = []
        if step is not None:
             if 'step' not in self.metrics_history:
                 self.metrics_history['step'] = []
             self.metrics_history['step'].append(step)

        for name, value in metrics.items():
            full_name = f"{prefix}{name}"
            
            # Store history
            if full_name not in self.metrics_history:
                self.metrics_history[full_name] = []
            self.metrics_history[full_name].append(value)
            
            # Prepare log message
            if isinstance(value, float):
                log_parts.append(f"{full_name}={value:.4f}")
            else:
                log_parts.append(f"{full_name}={value}")
        
        log_message = f"Metrics (Step {step}): " + " | ".join(log_parts) if step is not None else "Metrics: " + " | ".join(log_parts)
        self.logger.info(log_message)

    def log_gpu_memory(self, step: Optional[int] = None) -> None:
        """Logs memory usage for all available GPUs."""
        if not self.track_gpu:
            return
            
        gpu_metrics = {}
        for i in range(torch.cuda.device_count()):
            # Convert to MB
            mem_alloc = torch.cuda.memory_allocated(i) / (1024 ** 2)
            mem_reserved = torch.cuda.memory_reserved(i) / (1024 ** 2)
            gpu_metrics[f'gpu{i}_alloc_mb'] = mem_alloc
            gpu_metrics[f'gpu{i}_reserved_mb'] = mem_reserved
        
        self.log_metrics(gpu_metrics, step=step, prefix="system/")

    def save_metrics_to_csv(self, filename: Optional[str] = None) -> str:
        """Saves all recorded metrics to a CSV file."""
        if not self.metrics_history:
            self.logger.warning("No metrics recorded, skipping CSV export.")
            return ""
        
        # Pad lists to ensure equal length
        max_len = max(len(v) for v in self.metrics_history.values())
        for k in self.metrics_history:
            self.metrics_history[k].extend([None] * (max_len - len(self.metrics_history[k])))

        try:
            df = pd.DataFrame(self.metrics_history)
            
            if filename is None:
                filename = f"{self.name}_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
            csv_path = self.log_dir / filename
            df.to_csv(csv_path, index=False)
            self.logger.info(f"Metrics saved to {csv_path}")
            return str(csv_path)
        except Exception as e:
            self.logger.error(f"Failed to save metrics to CSV: {e}")
            return ""

    def close(self) -> None:
        """Safely closes all file handlers."""
        self.logger.info("Closing logger.")
        handlers = self.logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.logger.removeHandler(handler)

# ====================
#  Independent Test Module
# ====================
if __name__ == '__main__':
    print("===== Running Test for utils/logger.py =====")
    
    # --- Test Setup ---
    TEST_LOG_DIR = "temp_test_logs"
    if os.path.exists(TEST_LOG_DIR):
        shutil.rmtree(TEST_LOG_DIR) # Clean up old test logs

    # 1. Init Logger
    print("\n[Test 1] Initializing logger...")
    logger = GraphNASLogger(log_dir=TEST_LOG_DIR, name="test_run", log_level="INFO")
    
    # 2. Test basic logging
    print("\n[Test 2] Testing basic log messages...")
    logger.logger.debug("This is a debug message. Should not be visible with INFO level.")
    logger.logger.info("This is an info message.")
    logger.logger.warning("This is a warning message.")
    logger.logger.error("This is an error message.")

    # 3. Simulate training loop and log metrics
    print("\n[Test 3] Simulating training loop and logging metrics...")
    for epoch in range(1, 4):
        # Mock train metrics
        train_metrics = {'loss': 1.5 - epoch * 0.4, 'acc': 0.75 + epoch * 0.05}
        logger.log_metrics(train_metrics, step=epoch, prefix='train/')

        # Mock validation metrics
        val_metrics = {'loss': 1.3 - epoch * 0.35, 'acc': 0.78 + epoch * 0.04}
        logger.log_metrics(val_metrics, step=epoch, prefix='val/')

        # Log GPU memory
        logger.log_gpu_memory(step=epoch)

    # 4. Save metrics to CSV
    print("\n[Test 4] Saving metrics to CSV file...")
    logger.save_metrics_to_csv()

    # 5. Close Logger
    logger.close()
    
    print(f"\n===== Test Finished. Check folder '{TEST_LOG_DIR}' for logs and csv file. =====")
