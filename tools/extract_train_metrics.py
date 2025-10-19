import argparse
import re
import json

def extract_from_log(path):
    patterns = {'dataset': re.compile('实验-训练数据集[:：]\\s*(\\S+)'), 'total_train_time_s': re.compile('(?:实验-训练时长|实验-纯训练时间)[:：]\\s*([0-9]*\\.?[0-9]+)s'), 'num_archs': re.compile('实验-训练架构数[:：]\\s*(\\d+)'), 'avg_time_per_arch_s': re.compile('(?:实验-平均单架训练时间|实验-平均架构训练时间)[:：]\\s*([0-9]*\\.?[0-9]+)s'), 'best_accuracy': re.compile('实验-最优架构精度[:：]\\s*([0-9]*\\.?[0-9]+)'), 'best_model_id': re.compile('实验-最优架构编号[:：]\\s*(\\S+)'), 'best_found_epoch': re.compile('实验-最优架构出现轮次[:：]\\s*(\\d+)')}
    out = {k: None for k in patterns.keys()}
    with open(path, 'r') as f:
        data = f.read()
    for k, p in patterns.items():
        m = p.search(data)
        if m:
            val = m.group(1)
            if k in ['num_archs', 'best_found_epoch']:
                out[k] = int(val)
            else:
                try:
                    out[k] = float(val)
                except Exception:
                    out[k] = val
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--out', required=False)
    args = parser.parse_args()
    metrics = extract_from_log(args.input)
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
    else:
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
if __name__ == '__main__':
    main()