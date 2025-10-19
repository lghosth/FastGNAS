import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
matplotlib.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei']
matplotlib.rcParams['axes.unicode_minus'] = False

def find_csv_file(log_dir: str) -> str:
    csv_path = os.path.join(log_dir, 'search_timeline.csv')
    if os.path.exists(csv_path):
        return csv_path
    raise FileNotFoundError(f'未找到CSV文件: {csv_path}')

def load_timeline_data(csv_file: str) -> pd.DataFrame:
    print(f'正在读取CSV文件: {csv_file}')
    df = pd.read_csv(csv_file)
    required_columns = ['cumulative_time', 'global_best_accuracy', 'sampled_arch_count']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f'CSV文件缺少必要的列: {missing_columns}')
    print(f'成功加载 {len(df)} 个数据点')
    print(f'数据列: {df.columns.tolist()}')
    return df

def plot_search_timeline(df: pd.DataFrame, output_dir: str=None, show: bool=True):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    ax1.plot(df['cumulative_time'], df['global_best_accuracy'], linewidth=2, color='#2E86AB', marker='o', markersize=3, label='全局最优准确率')
    final_time = df['cumulative_time'].iloc[-1]
    final_acc = df['global_best_accuracy'].iloc[-1]
    ax1.scatter([final_time], [final_acc], color='red', s=100, zorder=5, label=f'最终最优: {final_acc:.4f}')
    ax1.annotate(f'{final_acc:.4f}', xy=(final_time, final_acc), xytext=(10, 10), textcoords='offset points', bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7), arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    ax1.set_xlabel('累计时间 (秒)', fontsize=12)
    ax1.set_ylabel('全局最优准确率', fontsize=12)
    ax1.set_title('架构搜索过程 - 全局最优准确率随时间变化', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=10)
    stats_text = f'数据点数: {len(df)}\n总耗时: {final_time:.2f}s\n最优准确率: {final_acc:.4f}'
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax2.plot(df['sampled_arch_count'], df['global_best_accuracy'], linewidth=2, color='#A23B72', marker='s', markersize=3, label='全局最优准确率')
    if 'current_round' in df.columns:
        new_arch_points = df[df['current_round'] == 1]
        if len(new_arch_points) > 0:
            ax2.scatter(new_arch_points['sampled_arch_count'], new_arch_points['global_best_accuracy'], color='orange', s=50, marker='^', zorder=5, label='新架构开始', alpha=0.6)
    final_arch_count = df['sampled_arch_count'].iloc[-1]
    ax2.scatter([final_arch_count], [final_acc], color='red', s=100, zorder=5, label=f'最终最优: {final_acc:.4f}')
    ax2.annotate(f'{final_acc:.4f}', xy=(final_arch_count, final_acc), xytext=(10, 10), textcoords='offset points', bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7), arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    ax2.set_xlabel('采样架构序号', fontsize=12)
    ax2.set_ylabel('全局最优准确率', fontsize=12)
    ax2.set_title('架构搜索过程 - 全局最优准确率随架构数量变化', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=10)
    total_archs = df['sampled_arch_count'].max()
    stats_text2 = f'总架构数: {total_archs}\n平均每架构时间: {final_time / total_archs:.2f}s\n最优准确率: {final_acc:.4f}'
    ax2.text(0.02, 0.98, stats_text2, transform=ax2.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.tight_layout()
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'search_timeline.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f'\n✅ 图形已保存到: {output_path}')
        pdf_path = os.path.join(output_dir, 'search_timeline.pdf')
        plt.savefig(pdf_path, bbox_inches='tight')
        print(f'✅ PDF已保存到: {pdf_path}')
    if show:
        print('\n📊 显示图形...')
        plt.show()
    plt.close()

def plot_detailed_timeline(df: pd.DataFrame, output_dir: str=None, show: bool=True):
    if 'current_accuracy' not in df.columns or 'current_round' not in df.columns:
        print('⚠️  CSV数据不包含详细轮次信息，跳过详细图绘制')
        return
    fig, ax = plt.subplots(figsize=(14, 8))
    unique_archs = df['sampled_arch_count'].unique()
    colors = plt.cm.tab20(range(len(unique_archs)))
    for i, arch_idx in enumerate(unique_archs):
        arch_data = df[df['sampled_arch_count'] == arch_idx]
        if len(arch_data) > 0:
            color = colors[i % len(colors)]
            ax.plot(arch_data['cumulative_time'], arch_data['current_accuracy'], marker='o', markersize=4, linewidth=1.5, alpha=0.7, label=f'架构 {arch_idx}', color=color)
    ax.plot(df['cumulative_time'], df['global_best_accuracy'], linewidth=3, color='black', linestyle='--', label='全局最优', zorder=100)
    ax.set_xlabel('累计时间 (秒)', fontsize=12)
    ax.set_ylabel('准确率', fontsize=12)
    ax.set_title('架构搜索详细轨迹 - 每个架构的训练过程', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 11:
        handles = handles[:10] + [handles[-1]]
        labels = labels[:10] + [labels[-1]]
    ax.legend(handles, labels, fontsize=9, loc='best', ncol=2)
    plt.tight_layout()
    if output_dir:
        output_path = os.path.join(output_dir, 'search_timeline_detailed.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f'✅ 详细图已保存到: {output_path}')
    if show:
        plt.show()
    plt.close()

def print_summary_statistics(df: pd.DataFrame):
    print('\n' + '=' * 60)
    print('📊 搜索过程统计摘要')
    print('=' * 60)
    final_time = df['cumulative_time'].iloc[-1]
    final_acc = df['global_best_accuracy'].iloc[-1]
    total_archs = df['sampled_arch_count'].max()
    print(f'总耗时: {final_time:.2f} 秒')
    print(f'总架构数: {total_archs}')
    print(f'平均每架构时间: {final_time / total_archs:.2f} 秒')
    print(f'最终最优准确率: {final_acc:.4f}')
    best_idx = df[df['global_best_accuracy'] == final_acc].index[0]
    best_time = df.loc[best_idx, 'cumulative_time']
    best_arch = df.loc[best_idx, 'sampled_arch_count']
    print(f'\n最优架构信息:')
    print(f'  - 出现时间: {best_time:.2f} 秒 ({best_time / final_time * 100:.1f}%)')
    print(f'  - 架构序号: {best_arch}')
    if 'current_arch_name' in df.columns:
        best_arch_name = df.loc[best_idx, 'current_arch_name']
        print(f'  - 架构名称: {best_arch_name}')
    initial_acc = df['global_best_accuracy'].iloc[0]
    improvement = final_acc - initial_acc
    print(f'\n准确率提升:')
    print(f'  - 初始准确率: {initial_acc:.4f}')
    print(f'  - 最终准确率: {final_acc:.4f}')
    print(f'  - 绝对提升: {improvement:.4f}')
    print(f'  - 相对提升: {improvement / initial_acc * 100:.2f}%')
    print('=' * 60)

def main():
    parser = argparse.ArgumentParser(description='绘制架构搜索时间线图', formatter_class=argparse.RawDescriptionHelpFormatter, epilog='\n示例:\n  # 通过日志目录\n  python utils/plot_search_timeline.py --log_dir experiment/Ca-rl/log/run_20251005_205249\n\n  # 直接指定CSV文件\n  python utils/plot_search_timeline.py --csv_file experiment/Ca-rl/log/run_20251005_205249/search_timeline.csv\n\n  # 不显示图形，只保存\n  python utils/plot_search_timeline.py --log_dir experiment/Ca-rl/log/run_20251005_205249 --no-show\n\n  # 指定输出目录\n  python utils/plot_search_timeline.py --csv_file search_timeline.csv --output_dir ./plots\n        ')
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--log_dir', type=str, help='日志目录路径（包含search_timeline.csv）')
    input_group.add_argument('--csv_file', type=str, help='直接指定CSV文件路径')
    parser.add_argument('--output_dir', type=str, default=None, help='图形输出目录（默认保存到CSV文件所在目录）')
    parser.add_argument('--no-show', action='store_true', help='不显示图形，仅保存文件')
    parser.add_argument('--detailed', action='store_true', help='绘制详细的每架构轨迹图')
    args = parser.parse_args()
    try:
        if args.log_dir:
            csv_file = find_csv_file(args.log_dir)
            default_output_dir = args.log_dir
        else:
            csv_file = args.csv_file
            default_output_dir = os.path.dirname(os.path.abspath(csv_file))
        output_dir = args.output_dir if args.output_dir else default_output_dir
        df = load_timeline_data(csv_file)
        print_summary_statistics(df)
        print('\n📈 正在绘制搜索时间线图...')
        plot_search_timeline(df, output_dir=output_dir, show=not args.no_show)
        if args.detailed:
            print('\n📈 正在绘制详细轨迹图...')
            plot_detailed_timeline(df, output_dir=output_dir, show=not args.no_show)
        print('\n✅ 完成!')
    except FileNotFoundError as e:
        print(f'\n❌ 错误: {e}')
        print('\n提示: 请确保指定的目录包含 search_timeline.csv 文件')
        return 1
    except Exception as e:
        print(f'\n❌ 发生错误: {e}')
        import traceback
        traceback.print_exc()
        return 1
    return 0
if __name__ == '__main__':
    exit(main())