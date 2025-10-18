from typing import List, Dict, Tuple

def evaluate_balance(architectures: List[Dict], 
                     num_workers: int, 
                     tolerance: float) -> Tuple[bool, float]:
    """
    Evaluates if a set of architectures with predicted loads is balanced using a bucket algorithm.

    Args:
        architectures (List[Dict]): A list of architectures, each with 'arch_spec', 'arch_id', and 'load'.
        num_workers (int): The number of workers (buckets).
        tolerance (float): The load imbalance tolerance (NGT).

    Returns:
        Tuple[bool, float]: A boolean indicating if the set is balanced, and the current max load difference.
    """
    if not architectures:
        return True, 0.0


    sorted_archs = sorted(architectures, key=lambda x: x['load'], reverse=True)

    buckets = [0.0] * num_workers

    for arch in sorted_archs:
        min_load_bucket_idx = buckets.index(min(buckets))
        buckets[min_load_bucket_idx] += arch['load']

    max_load = max(buckets)
    min_load = min(buckets)
    
    # The load difference is defined here as the absolute difference
    load_difference = max_load - min_load

    # Here, we assume tolerance is an absolute value (NGT) in the same units as the load.
    # If NGT were a percentage, the check would be: 
    # max_load - min_load <= (sum(buckets)/num_workers) * tolerance
    
    is_balanced = load_difference <= tolerance

    return is_balanced, load_difference
