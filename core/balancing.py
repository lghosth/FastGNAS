from typing import List, Dict, Tuple

def evaluate_balance(architectures: List[Dict], 
                     num_workers: int, 
                     tolerance: float) -> Tuple[bool, float]:
    """
    使用桶算法评估一组带有负载的架构是否均衡。

    Args:
        architectures (List[Dict]): 架构列表, 每个dict包含 'arch_spec', 'arch_id', 'load'
        num_workers (int): 工作单元（桶）的数量
        tolerance (float): 负载不均衡容忍度 (NGT)

    Returns:
        Tuple[bool, float]: 是否满足均衡条件, 以及当前的最大负载差
    """
    if not architectures:
        return True, 0.0

    # 1. 按负载降序排序
    sorted_archs = sorted(architectures, key=lambda x: x['load'], reverse=True)

    # 2. 初始化M个桶
    buckets = [0.0] * num_workers

    # 3. 贪心分配：将架构分配给当前负载最小的桶
    for arch in sorted_archs:
        min_load_bucket_idx = buckets.index(min(buckets))
        buckets[min_load_bucket_idx] += arch['load']

    # 4. 判断是否满足均衡条件
    max_load = max(buckets)
    min_load = min(buckets)
    
    # 负载差异可以定义为绝对值或相对值，这里使用绝对值
    load_difference = max_load - min_load

    # 容忍度可以解释为绝对值或相对于平均负载的百分比
    # 这里我们假设tolerance是与负载同量纲的绝对值 NGT
    # 如果 NGT 是百分比, 则应为: max_load - min_load <= (sum(buckets)/num_workers) * tolerance
    
    is_balanced = load_difference <= tolerance

    return is_balanced, load_difference