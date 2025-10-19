from typing import List, Dict, Tuple

def evaluate_balance(architectures: List[Dict], num_workers: int, tolerance: float) -> Tuple[bool, float]:
    if not architectures:
        return (True, 0.0)
    sorted_archs = sorted(architectures, key=lambda x: x['load'], reverse=True)
    buckets = [0.0] * num_workers
    for arch in sorted_archs:
        min_load_bucket_idx = buckets.index(min(buckets))
        buckets[min_load_bucket_idx] += arch['load']
    max_load = max(buckets)
    min_load = min(buckets)
    average_load = sum(buckets) / num_workers
    load_difference = max_load - min_load
    tolerance_absolute = average_load * tolerance
    is_balanced = load_difference <= tolerance_absolute
    return (is_balanced, load_difference)

def assign_to_buckets(architectures: List[Dict], num_workers: int) -> Tuple[List[List[Dict]], List[float]]:
    if not architectures:
        return ([[] for _ in range(num_workers)], [0.0] * num_workers)
    sorted_archs = sorted(architectures, key=lambda x: x['load'], reverse=True)
    buckets = [[] for _ in range(num_workers)]
    bucket_loads = [0.0] * num_workers
    for arch in sorted_archs:
        min_load_bucket_idx = bucket_loads.index(min(bucket_loads))
        buckets[min_load_bucket_idx].append(arch)
        bucket_loads[min_load_bucket_idx] += arch['load']
    return (buckets, bucket_loads)

def get_bucket_assignment_summary(buckets: List[List[Dict]], bucket_loads: List[float]) -> Dict[str, any]:
    summary = {'num_buckets': len(buckets), 'total_architectures': sum((len(bucket) for bucket in buckets)), 'bucket_details': [], 'load_statistics': {}}
    total_load = sum(bucket_loads)
    avg_load = total_load / len(bucket_loads) if bucket_loads else 0
    max_load = max(bucket_loads) if bucket_loads else 0
    min_load = min(bucket_loads) if bucket_loads else 0
    summary['load_statistics'] = {'total_load': total_load, 'average_load': avg_load, 'max_load': max_load, 'min_load': min_load, 'load_difference': max_load - min_load}
    for i, (bucket, load) in enumerate(zip(buckets, bucket_loads)):
        bucket_detail = {'bucket_id': i, 'architectures': [arch['arch_id'] for arch in bucket], 'architecture_count': len(bucket), 'total_load': load, 'load_percentage': load / total_load * 100 if total_load > 0 else 0}
        summary['bucket_details'].append(bucket_detail)
    return summary