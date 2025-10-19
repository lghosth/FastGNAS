import os
import logging
import shutil
from typing import Dict, Any, Optional
import torch
from datetime import datetime
import pandas as pd
from pathlib import Path

class GraphNASLogger:

    def __init__(self, log_dir: str='logs/exp', name: str='graphnas_run', log_level: str='INFO', track_gpu: bool=True):
        self.log_dir = Path(log_dir)
        self.name = name
        self.metrics_history = {}
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level.upper())
        self.logger.propagate = False
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        date_fmt = '%Y-%m-%d %H:%M:%S'
        formatter = logging.Formatter(fmt=log_fmt, datefmt=date_fmt)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file_path = self.log_dir / f'{self.name}_{timestamp}.log'
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.logger.info(f"Logger '{self.name}' initialized. Logging to file: {log_file_path}")
        self.track_gpu = track_gpu and torch.cuda.is_available()
        if track_gpu and (not self.track_gpu):
            self.logger.warning('GPU tracking was requested, but CUDA is not available.')
        elif self.track_gpu:
            self.logger.info(f'GPU memory tracking enabled for {torch.cuda.device_count()} device(s).')

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int]=None, prefix: str='') -> None:
        log_parts = []
        if step is not None:
            if 'step' not in self.metrics_history:
                self.metrics_history['step'] = []
            self.metrics_history['step'].append(step)
        for name, value in metrics.items():
            full_name = f'{prefix}{name}'
            if full_name not in self.metrics_history:
                self.metrics_history[full_name] = []
            self.metrics_history[full_name].append(value)
            if isinstance(value, float):
                log_parts.append(f'{full_name}={value:.4f}')
            else:
                log_parts.append(f'{full_name}={value}')
        log_message = f'Metrics (Step {step}): ' + ' | '.join(log_parts) if step is not None else 'Metrics: ' + ' | '.join(log_parts)
        self.logger.info(log_message)

    def log_gpu_memory(self, step: Optional[int]=None) -> None:
        if not self.track_gpu:
            return
        gpu_metrics = {}
        for i in range(torch.cuda.device_count()):
            mem_alloc = torch.cuda.memory_allocated(i) / 1024 ** 2
            mem_reserved = torch.cuda.memory_reserved(i) / 1024 ** 2
            gpu_metrics[f'gpu{i}_alloc_mb'] = mem_alloc
            gpu_metrics[f'gpu{i}_reserved_mb'] = mem_reserved
        self.log_metrics(gpu_metrics, step=step, prefix='system/')

    def save_metrics_to_csv(self, filename: Optional[str]=None) -> None:
        if not self.metrics_history:
            self.logger.warning('No metrics have been recorded, skipping CSV export.')
            return
        max_len = max((len(v) for v in self.metrics_history.values()))
        for k in self.metrics_history:
            self.metrics_history[k].extend([None] * (max_len - len(self.metrics_history[k])))
        try:
            df = pd.DataFrame(self.metrics_history)
            if filename is None:
                filename = f"{self.name}_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            csv_path = self.log_dir / filename
            df.to_csv(csv_path, index=False)
            self.logger.info(f'Metrics successfully saved to {csv_path}')
        except Exception as e:
            self.logger.error(f'Failed to save metrics to CSV: {e}')

    def save_metrics_to_excel(self, filename: Optional[str]=None, save_csv: bool=True) -> None:
        if not self.metrics_history:
            self.logger.warning('No metrics have been recorded, skipping export.')
            return
        max_len = max((len(v) for v in self.metrics_history.values()))
        for k in self.metrics_history:
            self.metrics_history[k].extend([None] * (max_len - len(self.metrics_history[k])))
        try:
            df = pd.DataFrame(self.metrics_history)
            if filename is None:
                base = f"{self.name}_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            else:
                base = str(filename)
            csv_path = self.log_dir / f'{base}.csv'
            xlsx_path = self.log_dir / f'{base}.xlsx'
            if save_csv:
                df.to_csv(csv_path, index=False)
                self.logger.info(f'Metrics CSV saved to {csv_path}')
            try:
                df.to_excel(xlsx_path, index=False)
                self.logger.info(f'Metrics Excel saved to {xlsx_path}')
            except Exception as e:
                self.logger.warning(f'Failed to save metrics to Excel: {e}. To enable Excel export, install openpyxl: pip install openpyxl')
        except Exception as e:
            self.logger.error(f'Failed to export metrics: {e}')

    def close(self) -> None:
        self.logger.info('Closing logger.')
        handlers = self.logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.logger.removeHandler(handler)
if __name__ == '__main__':
    print('===== Running Test for utils/logger.py =====')
    TEST_LOG_DIR = 'temp_test_logs'
    if os.path.exists(TEST_LOG_DIR):
        shutil.rmtree(TEST_LOG_DIR)
    print('\n[Test 1] Initializing logger...')
    logger = GraphNASLogger(log_dir=TEST_LOG_DIR, name='test_run', log_level='INFO')
    print('\n[Test 2] Testing basic log messages...')
    logger.logger.debug('This is a debug message. Should not be visible with INFO level.')
    logger.logger.info('This is an info message.')
    logger.logger.warning('This is a warning message.')
    logger.logger.error('This is an error message.')
    print('\n[Test 3] Simulating training loop and logging metrics...')
    for epoch in range(1, 4):
        train_metrics = {'loss': 1.5 - epoch * 0.4, 'acc': 0.75 + epoch * 0.05}
        logger.log_metrics(train_metrics, step=epoch, prefix='train/')
        val_metrics = {'loss': 1.3 - epoch * 0.35, 'acc': 0.78 + epoch * 0.04}
        logger.log_metrics(val_metrics, step=epoch, prefix='val/')
        logger.log_gpu_memory(step=epoch)
    print('\n[Test 4] Saving metrics to CSV file...')
    logger.save_metrics_to_csv()
    logger.close()
    print(f"\n===== Test Finished. Check the folder '{TEST_LOG_DIR}' for logs and csv file. =====")