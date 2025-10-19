import os
import logging
from typing import List, Dict, Optional
import pandas as pd
logger = logging.getLogger('export_timeline')

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def normalize_and_save_timeline(global_timeline: List[Dict], log_dir: str, filename_base: str='search_timeline', save_excel: bool=True, excel_row_threshold: int=10000) -> Dict[str, Optional[str]]:
    _ensure_dir(log_dir)
    columns = ['cumulative_time', 'global_best_accuracy', 'sampled_arch_count', 'best_arch_index', 'current_arch_name', 'current_round', 'current_accuracy']
    try:
        df = pd.DataFrame(global_timeline)
    except Exception as e:
        logger.error(f'无法将 global_timeline 转为 DataFrame: {e}')
        raise
    for col in columns:
        if col not in df.columns:
            df[col] = None
    ordered_cols = columns + [c for c in df.columns.tolist() if c not in columns]
    df = df[ordered_cols]
    csv_path = os.path.join(log_dir, f'{filename_base}.csv')
    try:
        df.to_csv(csv_path, index=False)
        logger.info(f'Saved timeline CSV to {csv_path}')
    except Exception as e:
        logger.error(f'Failed to save CSV: {e}')
        raise
    xlsx_path = None
    if save_excel and len(df) <= excel_row_threshold:
        try:
            xlsx_path = os.path.join(log_dir, f'{filename_base}.xlsx')
            df.to_excel(xlsx_path, index=False)
            logger.info(f'Saved timeline Excel to {xlsx_path}')
        except Exception as e:
            logger.warning(f'Could not save Excel ({e}), CSV saved at {csv_path}. To enable Excel export, install openpyxl: pip install openpyxl')
            xlsx_path = None
    elif save_excel:
        logger.info(f'Skipping Excel export due to row count ({len(df)} > {excel_row_threshold})')
    return {'csv': csv_path, 'xlsx': xlsx_path}