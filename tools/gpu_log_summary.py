import argparse
import csv
import json
import numpy as np
from collections import defaultdict

def summarize_csv(path):
    per_gpu = defaultdict(list)
    timestamps = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                idx = int(row['index'])
                util = float(row['utilization.gpu'])
                mem_used = float(row['memory.used'])
            except Exception:
                continue
            per_gpu[idx].append({'util': util, 'mem': mem_used})
            timestamps.append(row['timestamp'])
    result = {'per_gpu': {}, 'aggregate': {}}
    all_utils = []
    for idx, samples in sorted(per_gpu.items()):
        utils = [s['util'] for s in samples]
        mems = [s['mem'] for s in samples]
        arr_utils = np.array(utils) if utils else np.array([0.0])
        arr_mems = np.array(mems) if mems else np.array([0.0])
        result['per_gpu'][str(idx)] = {'util_peak': float(np.max(arr_utils)), 'util_min': float(np.min(arr_utils)), 'util_mean': float(np.mean(arr_utils)), 'util_median': float(np.median(arr_utils)), 'mem_peak': float(np.max(arr_mems)), 'mem_min': float(np.min(arr_mems)), 'mem_mean': float(np.mean(arr_mems)), 'mem_median': float(np.median(arr_mems)), 'samples': len(arr_utils)}
        all_utils.append(arr_utils)
    if all_utils:
        maxlen = max((a.shape[0] for a in all_utils))
        stacked = np.zeros((len(all_utils), maxlen), dtype=float)
        for i, a in enumerate(all_utils):
            stacked[i, :a.shape[0]] = a
        mean_per_t = np.mean(stacked, axis=0)
        result['aggregate'] = {'util_peak': float(np.max(mean_per_t)), 'util_min': float(np.min(mean_per_t)), 'util_mean': float(np.mean(mean_per_t)), 'util_median': float(np.median(mean_per_t)), 'sample_points': int(mean_per_t.shape[0])}
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--out', required=False)
    args = parser.parse_args()
    summary = summarize_csv(args.input)
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    else:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
if __name__ == '__main__':
    main()