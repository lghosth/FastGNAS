import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_memory_usage(csv_file: str, output_file: str=None):
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f'错误: 无法读取CSV文件 - {e}')
        return
    required_cols = ['elapsed_time', 'cpu_used_gb', 'cpu_percent']
    if not all((col in df.columns for col in required_cols)):
        print(f'错误: CSV文件缺少必要的列')
        print(f'需要: {required_cols}')
        print(f'实际: {list(df.columns)}')
        return
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle('FastGNAS 内存使用监控', fontsize=16, fontweight='bold')
    ax1 = axes[0]
    ax1.plot(df['elapsed_time'], df['cpu_used_gb'], linewidth=2, color='#2E86AB', label='已使用')
    ax1.fill_between(df['elapsed_time'], 0, df['cpu_used_gb'], alpha=0.3, color='#2E86AB')
    max_idx = df['cpu_used_gb'].idxmax()
    max_val = df['cpu_used_gb'].max()
    max_time = df.loc[max_idx, 'elapsed_time']
    ax1.plot(max_time, max_val, 'ro', markersize=8)
    ax1.annotate(f'峰值: {max_val:.2f}GB', xy=(max_time, max_val), xytext=(10, 10), textcoords='offset points', bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7), arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    ax1.set_xlabel('运行时间 (秒)', fontsize=12)
    ax1.set_ylabel('CPU内存使用量 (GB)', fontsize=12)
    ax1.set_title('CPU内存使用趋势', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax2 = axes[1]
    ax2.plot(df['elapsed_time'], df['cpu_percent'], linewidth=2, color='#A23B72', label='使用率')
    ax2.fill_between(df['elapsed_time'], 0, df['cpu_percent'], alpha=0.3, color='#A23B72')
    ax2.axhline(y=80, color='orange', linestyle='--', linewidth=1.5, label='警戒线 (80%)')
    ax2.axhline(y=90, color='red', linestyle='--', linewidth=1.5, label='危险线 (90%)')
    ax2.set_xlabel('运行时间 (秒)', fontsize=12)
    ax2.set_ylabel('CPU内存使用率 (%)', fontsize=12)
    ax2.set_title('CPU内存使用率趋势', fontsize=14)
    ax2.set_ylim([0, 100])
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    stats_text = f"统计信息:\n初始: {df['cpu_used_gb'].iloc[0]:.2f}GB\n最终: {df['cpu_used_gb'].iloc[-1]:.2f}GB\n峰值: {max_val:.2f}GB\n增长: {df['cpu_used_gb'].iloc[-1] - df['cpu_used_gb'].iloc[0]:.2f}GB\n持续: {df['elapsed_time'].iloc[-1]:.1f}s"
    fig.text(0.02, 0.02, stats_text, fontsize=9, family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f'✓ 图表已保存到: {output_file}')
    else:
        csv_path = Path(csv_file)
        output_file = csv_path.parent / f'{csv_path.stem}_plot.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f'✓ 图表已保存到: {output_file}')
    try:
        plt.show()
    except:
        pass

def main():
    parser = argparse.ArgumentParser(description='FastGNAS 内存使用曲线绘制工具')
    parser.add_argument('csv_file', type=str, help='内存监控CSV文件路径')
    parser.add_argument('--output', '-o', type=str, default=None, help='输出图片文件路径（默认自动生成）')
    args = parser.parse_args()
    if not Path(args.csv_file).exists():
        print(f'错误: 文件不存在 - {args.csv_file}')
        return 1
    plot_memory_usage(args.csv_file, args.output)
    return 0
if __name__ == '__main__':
    exit(main())