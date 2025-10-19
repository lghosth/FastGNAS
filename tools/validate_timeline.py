import os
import argparse
import pandas as pd
REQUIRED_COLUMNS = ['cumulative_time', 'global_best_accuracy', 'sampled_arch_count', 'best_arch_index', 'current_arch_name', 'current_round', 'current_accuracy']

def validate(log_dir: str) -> int:
    csv_path = os.path.join(log_dir, 'search_timeline.csv')
    xlsx_path = os.path.join(log_dir, 'search_timeline.xlsx')
    if not os.path.exists(csv_path):
        print(f'Missing CSV: {csv_path}')
        return 1
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        print(f'CSV missing columns: {missing}')
        return 2
    print(f'CSV OK: {csv_path} (rows={len(df)})')
    if os.path.exists(xlsx_path):
        try:
            df_x = pd.read_excel(xlsx_path)
            missing_x = [c for c in REQUIRED_COLUMNS if c not in df_x.columns]
            if missing_x:
                print(f'XLSX missing columns: {missing_x}')
                return 3
            print(f'XLSX OK: {xlsx_path} (rows={len(df_x)})')
        except Exception as e:
            print(f'Failed to read xlsx: {e}')
            return 4
    else:
        print(f'XLSX not found (optional): {xlsx_path}')
    return 0

def main():
    parser = argparse.ArgumentParser(description='Validate search_timeline CSV/XLSX files in a log directory')
    parser.add_argument('log_dir', type=str, help='实验日志目录')
    args = parser.parse_args()
    exit_code = validate(args.log_dir)
    raise SystemExit(exit_code)
if __name__ == '__main__':
    main()