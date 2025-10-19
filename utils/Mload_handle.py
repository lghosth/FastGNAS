import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any
import hashlib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Noto Sans CJK TC', 'AR PL UMing CN', 'AR PL UKai CN', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class MloadDataAnalyzer:

    def __init__(self, data_file: str='dataset/Mload_dataset/test.yaml'):
        self.data_file = data_file
        self.samples = []
        self.labels = []
        self.duplicate_groups = {}
        self.stats = {}

    def load_data(self) -> bool:
        try:
            print(f'正在加载数据文件: {self.data_file}')
            if not os.path.exists(self.data_file):
                print(f'错误: 文件不存在 {self.data_file}')
                return False
            with open(self.data_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            if 'samples' not in data or 'labels' not in data:
                print('错误: 数据文件格式不正确，缺少samples或labels字段')
                return False
            self.samples = data['samples']
            self.labels = data['labels']
            print(f'✅ 数据加载成功!')
            print(f'   - 样本数量: {len(self.samples)}')
            print(f'   - 标签数量: {len(self.labels)}')
            if len(self.samples) != len(self.labels):
                print(f'⚠️  警告: 样本数量({len(self.samples)})与标签数量({len(self.labels)})不匹配')
            return True
        except Exception as e:
            print(f'❌ 数据加载失败: {e}')
            return False

    def generate_architecture_hash(self, arch: List[Dict]) -> str:
        arch_str = json.dumps(arch, sort_keys=True)
        return hashlib.md5(arch_str.encode()).hexdigest()

    def detect_duplicates(self) -> Dict[str, List[int]]:
        print('🔍 正在检测重复的模型架构...')
        arch_hash_map = defaultdict(list)
        for idx, sample in enumerate(self.samples):
            arch = sample.get('architecture', [])
            arch_hash = self.generate_architecture_hash(arch)
            arch_hash_map[arch_hash].append(idx)
        self.duplicate_groups = {arch_hash: indices for arch_hash, indices in arch_hash_map.items() if len(indices) > 1}
        print(f'✅ 重复检测完成!')
        print(f'   - 唯一架构数量: {len(arch_hash_map)}')
        print(f'   - 重复架构组数: {len(self.duplicate_groups)}')
        total_duplicates = sum((len(indices) for indices in self.duplicate_groups.values()))
        print(f'   - 重复样本总数: {total_duplicates}')
        return self.duplicate_groups

    def analyze_duplicate_labels(self) -> Dict[str, Any]:
        print('📊 正在分析重复架构的标签差异...')
        duplicate_analysis = {'groups': [], 'summary': {'total_groups': len(self.duplicate_groups), 'max_duplicates': 0, 'min_duplicates': float('inf'), 'avg_duplicates': 0, 'label_variance_stats': []}}
        all_label_diffs = []
        for arch_hash, indices in self.duplicate_groups.items():
            group_labels = [self.labels[i] for i in indices]
            group_analysis = {'arch_hash': arch_hash, 'indices': indices, 'labels': group_labels, 'label_stats': {'mean': np.mean(group_labels), 'std': np.std(group_labels), 'min': np.min(group_labels), 'max': np.max(group_labels), 'range': np.max(group_labels) - np.min(group_labels)}}
            if len(group_labels) > 1:
                label_diffs = []
                for i in range(len(group_labels)):
                    for j in range(i + 1, len(group_labels)):
                        diff = abs(group_labels[i] - group_labels[j])
                        label_diffs.append(diff)
                        all_label_diffs.append(diff)
                group_analysis['label_diffs'] = label_diffs
                group_analysis['max_diff'] = max(label_diffs) if label_diffs else 0
                group_analysis['avg_diff'] = np.mean(label_diffs) if label_diffs else 0
            duplicate_analysis['groups'].append(group_analysis)
            duplicate_analysis['summary']['max_duplicates'] = max(duplicate_analysis['summary']['max_duplicates'], len(indices))
            duplicate_analysis['summary']['min_duplicates'] = min(duplicate_analysis['summary']['min_duplicates'], len(indices))
        if self.duplicate_groups:
            duplicate_analysis['summary']['avg_duplicates'] = np.mean([len(indices) for indices in self.duplicate_groups.values()])
            duplicate_analysis['summary']['min_duplicates'] = min([len(indices) for indices in self.duplicate_groups.values()])
        if all_label_diffs:
            duplicate_analysis['summary']['label_variance_stats'] = {'mean_diff': np.mean(all_label_diffs), 'std_diff': np.std(all_label_diffs), 'max_diff': np.max(all_label_diffs), 'min_diff': np.min(all_label_diffs), 'median_diff': np.median(all_label_diffs)}
        print(f'✅ 重复标签分析完成!')
        if all_label_diffs:
            print(f'   - 标签差异统计:')
            print(f'     * 平均差异: {np.mean(all_label_diffs):.6f}')
            print(f'     * 最大差异: {np.max(all_label_diffs):.6f}')
            print(f'     * 标准差: {np.std(all_label_diffs):.6f}')
        return duplicate_analysis

    def calculate_label_statistics(self) -> Dict[str, Any]:
        print('📈 正在计算标签统计信息...')
        labels_array = np.array(self.labels)
        basic_stats = {'count': len(labels_array), 'mean': np.mean(labels_array), 'median': np.median(labels_array), 'std': np.std(labels_array), 'var': np.var(labels_array), 'min': np.min(labels_array), 'max': np.max(labels_array), 'range': np.max(labels_array) - np.min(labels_array)}
        quantiles = {'q25': np.percentile(labels_array, 25), 'q50': np.percentile(labels_array, 50), 'q75': np.percentile(labels_array, 75), 'q90': np.percentile(labels_array, 90), 'q95': np.percentile(labels_array, 95), 'q99': np.percentile(labels_array, 99)}
        label_counts = Counter(self.labels)
        mode_info = {'mode': label_counts.most_common(1)[0][0] if label_counts else None, 'mode_frequency': label_counts.most_common(1)[0][1] if label_counts else 0, 'unique_values': len(label_counts), 'most_common_5': label_counts.most_common(5)}
        from scipy import stats
        try:
            skewness = stats.skew(labels_array)
            kurtosis = stats.kurtosis(labels_array)
        except:
            skewness = kurtosis = None
        self.stats = {'basic': basic_stats, 'quantiles': quantiles, 'mode': mode_info, 'shape': {'skewness': skewness, 'kurtosis': kurtosis}}
        print(f'✅ 标签统计计算完成!')
        print(f"   - 平均值: {basic_stats['mean']:.6f}")
        print(f"   - 中位数: {basic_stats['median']:.6f}")
        print(f"   - 标准差: {basic_stats['std']:.6f}")
        print(f"   - 最小值: {basic_stats['min']:.6f}")
        print(f"   - 最大值: {basic_stats['max']:.6f}")
        print(f"   - 众数: {mode_info['mode']:.6f} (出现{mode_info['mode_frequency']}次)")
        return self.stats

    def create_visualizations(self, output_dir: str='logs/mload_analysis') -> None:
        print(f'📊 正在生成统计图表，输出目录: {output_dir}')
        os.makedirs(output_dir, exist_ok=True)
        plt.style.use('seaborn-v0_8')
        fig_size = (12, 8)
        plt.figure(figsize=fig_size)
        plt.hist(self.labels, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(self.stats['basic']['mean'], color='red', linestyle='--', label=f"平均值: {self.stats['basic']['mean']:.4f}")
        plt.axvline(self.stats['basic']['median'], color='green', linestyle='--', label=f"中位数: {self.stats['basic']['median']:.4f}")
        plt.xlabel('训练时间 (秒)')
        plt.ylabel('频数')
        plt.title('训练时间标签分布直方图')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/label_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        plt.figure(figsize=(8, 6))
        plt.boxplot(self.labels, patch_artist=True, boxprops=dict(facecolor='lightblue', alpha=0.7))
        plt.ylabel('训练时间 (秒)')
        plt.title('训练时间标签箱线图')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/label_boxplot.png', dpi=300, bbox_inches='tight')
        plt.close()
        plt.figure(figsize=fig_size)
        plt.hist(self.labels, bins=50, density=True, alpha=0.7, color='lightgreen', edgecolor='black', label='直方图')
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(self.labels)
        x_range = np.linspace(min(self.labels), max(self.labels), 200)
        plt.plot(x_range, kde(x_range), 'r-', linewidth=2, label='核密度估计')
        plt.xlabel('训练时间 (秒)')
        plt.ylabel('密度')
        plt.title('训练时间标签密度分布')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/label_density.png', dpi=300, bbox_inches='tight')
        plt.close()
        if self.duplicate_groups:
            duplicate_analysis = self.analyze_duplicate_labels()
            plt.figure(figsize=fig_size)
            group_sizes = [len(indices) for indices in self.duplicate_groups.values()]
            plt.hist(group_sizes, bins=max(group_sizes) - 1, alpha=0.7, color='orange', edgecolor='black')
            plt.xlabel('重复组大小')
            plt.ylabel('组数量')
            plt.title('重复架构组大小分布')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/duplicate_group_sizes.png', dpi=300, bbox_inches='tight')
            plt.close()
            all_diffs = []
            for group in duplicate_analysis['groups']:
                if 'label_diffs' in group:
                    all_diffs.extend(group['label_diffs'])
            if all_diffs:
                plt.figure(figsize=fig_size)
                plt.hist(all_diffs, bins=30, alpha=0.7, color='purple', edgecolor='black')
                plt.xlabel('标签差异 (绝对值)')
                plt.ylabel('频数')
                plt.title('重复架构标签差异分布')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(f'{output_dir}/label_differences.png', dpi=300, bbox_inches='tight')
                plt.close()
        self._plot_architecture_features(output_dir)
        self._plot_temporal_analysis(output_dir)
        print(f'✅ 所有图表已保存到: {output_dir}')

    def analyze_same_arch_label_differences(self, output_dir: str='logs/mload_analysis', top_k: int=20) -> Dict[str, Any]:
        print('📈 正在分析相同架构的标签差值...')
        os.makedirs(output_dir, exist_ok=True)
        if not self.samples or not self.labels:
            if not self.load_data():
                return {}
        if not self.duplicate_groups:
            self.detect_duplicates()
        per_arch_stats: List[Dict[str, Any]] = []
        all_diffs: List[float] = []
        for arch_hash, indices in self.duplicate_groups.items():
            if len(indices) <= 1:
                continue
            arch_labels = [self.labels[i] for i in indices]
            pairwise_diffs: List[float] = []
            for i in range(len(arch_labels)):
                for j in range(i + 1, len(arch_labels)):
                    pairwise_diffs.append(abs(arch_labels[i] - arch_labels[j]))
            if not pairwise_diffs:
                continue
            all_diffs.extend(pairwise_diffs)
            per_arch_stats.append({'arch_hash': arch_hash, 'group_size': len(indices), 'num_pairs': len(pairwise_diffs), 'mean_diff': float(np.mean(pairwise_diffs)), 'std_diff': float(np.std(pairwise_diffs)), 'min_diff': float(np.min(pairwise_diffs)), 'max_diff': float(np.max(pairwise_diffs)), 'range_diff': float(np.max(pairwise_diffs) - np.min(pairwise_diffs))})
        import pandas as pd
        if per_arch_stats:
            df = pd.DataFrame(per_arch_stats)
        else:
            empty_cols = ['arch_hash', 'group_size', 'num_pairs', 'mean_diff', 'std_diff', 'min_diff', 'max_diff', 'range_diff']
            df = pd.DataFrame({col: [] for col in empty_cols})
        csv_path = os.path.join(output_dir, 'duplicate_label_diff_stats.csv')
        df.to_csv(csv_path, index=False)
        if all_diffs:
            plt.figure(figsize=(12, 8))
            plt.hist(all_diffs, bins=50, alpha=0.75, color='steelblue', edgecolor='black')
            plt.xlabel('绝对标签差异')
            plt.ylabel('频数')
            plt.title('重复架构标签差异分布')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            hist_path = os.path.join(output_dir, 'dup_label_diff_hist.png')
            plt.savefig(hist_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            hist_path = None
        if not df.empty:
            df_sorted_std = df.sort_values(by='std_diff', ascending=False).head(top_k)
            plt.figure(figsize=(14, 7))
            sns.barplot(x='arch_hash', y='std_diff', data=df_sorted_std, palette='viridis')
            plt.xticks(rotation=75, ha='right')
            plt.xlabel('架构哈希值 (按标签差异标准差Top-K)')
            plt.ylabel('成对标签差异标准差')
            plt.title(f'按标签差异标准差排名的前{top_k}个架构')
            plt.tight_layout()
            topk_std_path = os.path.join(output_dir, f'dup_label_diff_top{top_k}_std.png')
            plt.savefig(topk_std_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            topk_std_path = None
        if not df.empty:
            plt.figure(figsize=(10, 7))
            sns.scatterplot(x='group_size', y='std_diff', data=df, alpha=0.7)
            plt.xlabel('重复组大小')
            plt.ylabel('成对标签差异标准差')
            plt.title('组大小与标签差异标准差关系')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            size_vs_std_path = os.path.join(output_dir, 'dup_label_diff_size_vs_std.png')
            plt.savefig(size_vs_std_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            size_vs_std_path = None
        summary = {'num_duplicate_groups': int(len(self.duplicate_groups)), 'num_groups_with_pairs': int(len(per_arch_stats)), 'global_stats': {'num_diffs': int(len(all_diffs)), 'mean_diff': float(np.mean(all_diffs)) if all_diffs else None, 'std_diff': float(np.std(all_diffs)) if all_diffs else None, 'min_diff': float(np.min(all_diffs)) if all_diffs else None, 'max_diff': float(np.max(all_diffs)) if all_diffs else None}, 'artifacts': {'csv': csv_path, 'hist': hist_path, 'topk_std': topk_std_path, 'size_vs_std': size_vs_std_path}}
        print('✅ 相同架构标签差值分析完成!')
        if summary['global_stats']['mean_diff'] is not None:
            print(f"   - 差值均值: {summary['global_stats']['mean_diff']:.6f}")
            print(f"   - 差值标准差: {summary['global_stats']['std_diff']:.6f}")
            print(f"   - 差值范围: {summary['global_stats']['min_diff']:.6f} ~ {summary['global_stats']['max_diff']:.6f}")
        print(f'   - 统计CSV: {csv_path}')
        if hist_path:
            print(f'   - 直方图: {hist_path}')
        if topk_std_path:
            print(f'   - Top-K标准差柱状图: {topk_std_path}')
        if size_vs_std_path:
            print(f'   - 组大小 vs 标准差 散点图: {size_vs_std_path}')
        return {'per_arch_stats': per_arch_stats, 'summary': summary}

    def summarize_remaining_after_duplicate_handling(self, output_dir: str='logs/mload_analysis', policy: str='keep_first') -> Dict[str, Any]:
        print('🧹 正在根据策略处理重复架构并统计剩余数据...')
        os.makedirs(output_dir, exist_ok=True)
        if not self.samples or not self.labels:
            if not self.load_data():
                return {}
        if not self.duplicate_groups:
            self.detect_duplicates()
        from collections import defaultdict
        arch_map = defaultdict(list)
        for idx, sample in enumerate(self.samples):
            arch = sample.get('architecture', [])
            h = self.generate_architecture_hash(arch)
            arch_map[h].append(idx)
        keep_indices: List[int] = []
        if policy == 'keep_first':
            for h, idxs in arch_map.items():
                keep_indices.append(min(idxs))
        elif policy == 'drop_all_duplicates':
            for h, idxs in arch_map.items():
                if len(idxs) == 1:
                    keep_indices.append(idxs[0])
        else:
            print(f"⚠️ 未知策略: {policy}，使用默认 'keep_first'")
            for h, idxs in arch_map.items():
                keep_indices.append(min(idxs))
        keep_indices = sorted(set(keep_indices))
        remaining_samples = [self.samples[i] for i in keep_indices]
        remaining_labels = [self.labels[i] for i in keep_indices]
        import numpy as np
        basic_stats = {'count': int(len(remaining_labels)), 'mean': float(np.mean(remaining_labels)) if remaining_labels else None, 'median': float(np.median(remaining_labels)) if remaining_labels else None, 'std': float(np.std(remaining_labels)) if remaining_labels else None, 'min': float(np.min(remaining_labels)) if remaining_labels else None, 'max': float(np.max(remaining_labels)) if remaining_labels else None}
        import pandas as pd
        export_rows = []
        for idx in keep_indices:
            arch = self.samples[idx].get('architecture', [])
            h = self.generate_architecture_hash(arch)
            export_rows.append({'index': idx, 'arch_hash': h, 'label': self.labels[idx]})
        df_remaining = pd.DataFrame(export_rows)
        csv_path = os.path.join(output_dir, f'remaining_after_{policy}.csv')
        df_remaining.to_csv(csv_path, index=False)
        hist_path = None
        if remaining_labels:
            plt.figure(figsize=(12, 8))
            plt.hist(remaining_labels, bins=50, alpha=0.75, color='teal', edgecolor='black')
            plt.xlabel('训练时间 (秒)')
            plt.ylabel('频数')
            plt.title(f'去重处理后的标签分布 ({policy})')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            hist_path = os.path.join(output_dir, f'remaining_after_{policy}_hist.png')
            plt.savefig(hist_path, dpi=300, bbox_inches='tight')
            plt.close()
        result = {'policy': policy, 'kept_indices': keep_indices, 'basic_stats': basic_stats, 'artifacts': {'csv': csv_path, 'hist': hist_path}}
        print('✅ 剩余数据统计完成!')
        print(f"   - 保留样本数: {basic_stats['count']}")
        if basic_stats['mean'] is not None:
            print(f"   - 标签均值: {basic_stats['mean']:.6f}")
            print(f"   - 标签标准差: {basic_stats['std']:.6f}")
            print(f"   - 范围: {basic_stats['min']:.6f} ~ {basic_stats['max']:.6f}")
        print(f'   - 导出CSV: {csv_path}')
        if hist_path:
            print(f'   - 直方图: {hist_path}')
        return result

    def _plot_architecture_features(self, output_dir: str) -> None:
        print('🏗️  正在分析架构特征...')
        activations = []
        attention_types = []
        aggregation_types = []
        attention_heads = []
        hidden_units = []
        for sample in self.samples:
            arch = sample.get('architecture', [])
            for layer in arch:
                activations.append(layer.get('activation', 'unknown'))
                attention_types.append(layer.get('attention_type', 'unknown'))
                aggregation_types.append(layer.get('aggregation_type', 'unknown'))
                attention_heads.append(layer.get('attention_heads', 0))
                hidden_units.append(layer.get('hidden_units', 0))
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        activation_counts = Counter(activations)
        axes[0, 0].bar(activation_counts.keys(), activation_counts.values(), color='lightcoral', alpha=0.7)
        axes[0, 0].set_title('激活函数分布')
        axes[0, 0].set_xlabel('激活函数')
        axes[0, 0].set_ylabel('使用次数')
        axes[0, 0].tick_params(axis='x', rotation=45)
        attention_counts = Counter(attention_types)
        axes[0, 1].bar(attention_counts.keys(), attention_counts.values(), color='lightblue', alpha=0.7)
        axes[0, 1].set_title('注意力类型分布')
        axes[0, 1].set_xlabel('注意力类型')
        axes[0, 1].set_ylabel('使用次数')
        axes[0, 1].tick_params(axis='x', rotation=45)
        aggregation_counts = Counter(aggregation_types)
        axes[1, 0].bar(aggregation_counts.keys(), aggregation_counts.values(), color='lightgreen', alpha=0.7)
        axes[1, 0].set_title('聚合类型分布')
        axes[1, 0].set_xlabel('聚合类型')
        axes[1, 0].set_ylabel('使用次数')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 1].hist(hidden_units, bins=20, alpha=0.7, color='gold', edgecolor='black')
        axes[1, 1].set_title('隐藏单元数分布')
        axes[1, 1].set_xlabel('隐藏单元数')
        axes[1, 1].set_ylabel('频数')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/architecture_features.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_temporal_analysis(self, output_dir: str) -> None:
        print('⏰ 正在分析时间序列...')
        timestamps = []
        labels_by_time = []
        for sample, label in zip(self.samples, self.labels):
            timestamp_str = sample.get('timestamp', '')
            if timestamp_str:
                try:
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    timestamps.append(timestamp)
                    labels_by_time.append(label)
                except:
                    continue
        if len(timestamps) < 2:
            print('⚠️  时间戳数据不足，跳过时间序列分析')
            return
        sorted_data = sorted(zip(timestamps, labels_by_time))
        timestamps, labels_by_time = zip(*sorted_data)
        plt.figure(figsize=(15, 6))
        plt.plot(timestamps, labels_by_time, 'o-', alpha=0.7, markersize=3)
        plt.xlabel('时间')
        plt.ylabel('训练时间 (秒)')
        plt.title('训练时间标签随时间变化')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/temporal_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def generate_report(self, output_dir: str='logs/mload_analysis') -> str:
        print('📝 正在生成分析报告...')
        os.makedirs(output_dir, exist_ok=True)
        report_path = f'{output_dir}/analysis_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('=' * 60 + '\n')
            f.write('Mload数据集分析报告\n')
            f.write('=' * 60 + '\n')
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f'数据文件: {self.data_file}\n\n')
            f.write('1. 数据概览\n')
            f.write('-' * 30 + '\n')
            f.write(f'样本总数: {len(self.samples)}\n')
            f.write(f'标签总数: {len(self.labels)}\n')
            f.write(f"唯一架构数: {len(set((self.generate_architecture_hash(sample.get('architecture', [])) for sample in self.samples)))}\n\n")
            f.write('2. 标签统计信息\n')
            f.write('-' * 30 + '\n')
            if self.stats:
                basic = self.stats['basic']
                quantiles = self.stats['quantiles']
                mode = self.stats['mode']
                f.write(f"平均值: {basic['mean']:.6f}\n")
                f.write(f"中位数: {basic['median']:.6f}\n")
                f.write(f"标准差: {basic['std']:.6f}\n")
                f.write(f"方差: {basic['var']:.6f}\n")
                f.write(f"最小值: {basic['min']:.6f}\n")
                f.write(f"最大值: {basic['max']:.6f}\n")
                f.write(f"范围: {basic['range']:.6f}\n")
                f.write(f"众数: {mode['mode']:.6f} (出现{mode['mode_frequency']}次)\n")
                f.write(f"唯一值数: {mode['unique_values']}\n\n")
                f.write('分位数:\n')
                f.write(f"  Q25: {quantiles['q25']:.6f}\n")
                f.write(f"  Q50: {quantiles['q50']:.6f}\n")
                f.write(f"  Q75: {quantiles['q75']:.6f}\n")
                f.write(f"  Q90: {quantiles['q90']:.6f}\n")
                f.write(f"  Q95: {quantiles['q95']:.6f}\n")
                f.write(f"  Q99: {quantiles['q99']:.6f}\n\n")
            f.write('3. 重复架构分析\n')
            f.write('-' * 30 + '\n')
            f.write(f'重复架构组数: {len(self.duplicate_groups)}\n')
            if self.duplicate_groups:
                total_duplicates = sum((len(indices) for indices in self.duplicate_groups.values()))
                f.write(f'重复样本总数: {total_duplicates}\n')
                f.write(f'重复率: {total_duplicates / len(self.samples) * 100:.2f}%\n\n')
                duplicate_analysis = self.analyze_duplicate_labels()
                if duplicate_analysis['summary']['label_variance_stats']:
                    diff_stats = duplicate_analysis['summary']['label_variance_stats']
                    f.write('重复架构标签差异统计:\n')
                    f.write(f"  平均差异: {diff_stats['mean_diff']:.6f}\n")
                    f.write(f"  最大差异: {diff_stats['max_diff']:.6f}\n")
                    f.write(f"  标准差: {diff_stats['std_diff']:.6f}\n")
                    f.write(f"  中位数差异: {diff_stats['median_diff']:.6f}\n\n")
            f.write('4. 架构特征统计\n')
            f.write('-' * 30 + '\n')
            activations = []
            attention_types = []
            aggregation_types = []
            attention_heads = []
            hidden_units = []
            for sample in self.samples:
                arch = sample.get('architecture', [])
                for layer in arch:
                    activations.append(layer.get('activation', 'unknown'))
                    attention_types.append(layer.get('attention_type', 'unknown'))
                    aggregation_types.append(layer.get('aggregation_type', 'unknown'))
                    attention_heads.append(layer.get('attention_heads', 0))
                    hidden_units.append(layer.get('hidden_units', 0))
            f.write('激活函数使用频率:\n')
            for activation, count in Counter(activations).most_common():
                f.write(f'  {activation}: {count}次\n')
            f.write('\n注意力类型使用频率:\n')
            for att_type, count in Counter(attention_types).most_common():
                f.write(f'  {att_type}: {count}次\n')
            f.write('\n聚合类型使用频率:\n')
            for agg_type, count in Counter(aggregation_types).most_common():
                f.write(f'  {agg_type}: {count}次\n')
            f.write(f'\n注意力头数统计:\n')
            f.write(f'  平均值: {np.mean(attention_heads):.2f}\n')
            f.write(f'  中位数: {np.median(attention_heads):.2f}\n')
            f.write(f'  最大值: {np.max(attention_heads)}\n')
            f.write(f'  最小值: {np.min(attention_heads)}\n')
            f.write(f'\n隐藏单元数统计:\n')
            f.write(f'  平均值: {np.mean(hidden_units):.2f}\n')
            f.write(f'  中位数: {np.median(hidden_units):.2f}\n')
            f.write(f'  最大值: {np.max(hidden_units)}\n')
            f.write(f'  最小值: {np.min(hidden_units)}\n')
        print(f'✅ 分析报告已保存到: {report_path}')
        return report_path

    def run_full_analysis(self, output_dir: str='logs/mload_analysis') -> None:
        print('🚀 开始Mload数据集完整分析...')
        print('=' * 50)
        if not self.load_data():
            return
        self.calculate_label_statistics()
        self.detect_duplicates()
        self.create_visualizations(output_dir)
        self.analyze_same_arch_label_differences(output_dir)
        self.summarize_remaining_after_duplicate_handling(output_dir)
        self.generate_report(output_dir)
        print('=' * 50)
        print('🎉 分析完成!')
        print(f'📁 所有结果已保存到: {output_dir}')

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Mload数据集分析工具')
    parser.add_argument('--data_file', type=str, default='dataset/Mload_dataset/test.yaml', help='数据文件路径')
    parser.add_argument('--output_dir', type=str, default='logs/mload_analysis', help='输出目录')
    args = parser.parse_args()
    analyzer = MloadDataAnalyzer(args.data_file)
    analyzer.run_full_analysis(args.output_dir)
if __name__ == '__main__':
    main()