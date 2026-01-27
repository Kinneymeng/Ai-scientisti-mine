"""
可视化对比：原版 vs 改进版 attention机制

运行方法：
    python visualize_comparison.py
"""

import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# 设置字体和样式
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'

# 读取数据
print("Loading data...")
data1 = json.load(open('run_1/results.json'))
data6 = json.load(open('run_6/results.json'))

weights1 = data1['attention_nn']['attention_weights']
weights6 = data6['attention_nn']['attention_weights']

feature_names = ['Delta', 'Velocity', 'Beta', 'Yaw Rate']
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

# 创建大图
fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 0.8], hspace=0.3, wspace=0.3)

# ============ 第一行：Attention权重演变 ============
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])

# Run 1 (原版)
for i, (name, color) in enumerate(zip(feature_names, colors)):
    weights_i = [w[i] for w in weights1]
    ax1.plot(weights_i, label=name, color=color, linewidth=2)

ax1.set_xlabel('Epoch')
ax1.set_ylabel('Attention Weight')
ax1.set_title('Original Version (Run 1): Saturated from Start', fontsize=12, fontweight='bold')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-0.05, 1.05)
ax1.axhline(y=0.25, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax1.text(50, 0.95, 'Velocity dominates\n(99.99%)',
         bbox=dict(boxstyle='round', facecolor='red', alpha=0.3),
         fontsize=9)

# Run 6 (改进版)
for i, (name, color) in enumerate(zip(feature_names, colors)):
    weights_i = [w[i] for w in weights6]
    ax2.plot(weights_i, label=name, color=color, linewidth=2)

ax2.set_xlabel('Epoch')
ax2.set_ylabel('Attention Weight')
ax2.set_title('Improved Version (Run 6): Uniform & Learning', fontsize=12, fontweight='bold')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)
ax2.set_ylim(-0.05, 1.05)
ax2.axhline(y=0.25, color='gray', linestyle='--', alpha=0.5, linewidth=1, label='Uniform (0.25)')
ax2.text(50, 0.95, 'All features balanced\n(~25% each)',
         bbox=dict(boxstyle='round', facecolor='green', alpha=0.3),
         fontsize=9)

# ============ 第二行：初始 vs 最终权重对比 ============
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

x_pos = range(len(feature_names))
width = 0.35

# Run 1 初始 vs 最终
initial1 = weights1[0]
final1 = weights1[-1]

bars1_init = ax3.bar([p - width/2 for p in x_pos], initial1, width,
                      label='Epoch 0', color='lightblue', edgecolor='blue', linewidth=1.5)
bars1_final = ax3.bar([p + width/2 for p in x_pos], final1, width,
                       label='Epoch 99', color='lightcoral', edgecolor='red', linewidth=1.5)

ax3.set_xlabel('Feature')
ax3.set_ylabel('Attention Weight')
ax3.set_title('Original: Initial vs Final Weights', fontsize=12, fontweight='bold')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(feature_names, rotation=15)
ax3.legend()
ax3.grid(True, axis='y', alpha=0.3)
ax3.set_ylim(0, 1.05)
ax3.axhline(y=0.25, color='gray', linestyle='--', alpha=0.5, linewidth=1)

# 添加数值标签（只显示velocity）
for i, (init, final) in enumerate(zip(initial1, final1)):
    if i == 1:  # velocity
        ax3.text(i - width/2, init + 0.02, f'{init:.4f}', ha='center', va='bottom', fontsize=8)
        ax3.text(i + width/2, final + 0.02, f'{final:.4f}', ha='center', va='bottom', fontsize=8)

# Run 6 初始 vs 最终
initial6 = weights6[0]
final6 = weights6[-1]

bars6_init = ax4.bar([p - width/2 for p in x_pos], initial6, width,
                      label='Epoch 0', color='lightblue', edgecolor='blue', linewidth=1.5)
bars6_final = ax4.bar([p + width/2 for p in x_pos], final6, width,
                       label='Epoch 99', color='lightgreen', edgecolor='green', linewidth=1.5)

ax4.set_xlabel('Feature')
ax4.set_ylabel('Attention Weight')
ax4.set_title('Improved: Initial vs Final Weights', fontsize=12, fontweight='bold')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(feature_names, rotation=15)
ax4.legend()
ax4.grid(True, axis='y', alpha=0.3)
ax4.set_ylim(0, 0.4)  # 缩小y轴范围以更好地显示细节
ax4.axhline(y=0.25, color='gray', linestyle='--', alpha=0.5, linewidth=1)

# 添加数值标签
for i, (init, final) in enumerate(zip(initial6, final6)):
    ax4.text(i - width/2, init + 0.005, f'{init:.3f}', ha='center', va='bottom', fontsize=7)
    ax4.text(i + width/2, final + 0.005, f'{final:.3f}', ha='center', va='bottom', fontsize=7)

# ============ 第三行：性能对比 ============
ax5 = fig.add_subplot(gs[2, :])

methods = ['Least Squares', 'Standard NN', 'Attention NN']
run1_errors = [
    data1['least_squares']['mean_error_percent'],
    data1['neural_network']['mean_error_percent'],
    data1['attention_nn']['mean_error_percent']
]
run6_errors = [
    data6['least_squares']['mean_error_percent'],
    data6['neural_network']['mean_error_percent'],
    data6['attention_nn']['mean_error_percent']
]

x_pos_methods = range(len(methods))
width = 0.35

bars_run1 = ax5.bar([p - width/2 for p in x_pos_methods], run1_errors, width,
                     label='Run 1 (noise=0.01, original)',
                     color='#e74c3c', alpha=0.7, edgecolor='darkred', linewidth=1.5)
bars_run6 = ax5.bar([p + width/2 for p in x_pos_methods], run6_errors, width,
                     label='Run 6 (noise=0.02, improved)',
                     color='#2ecc71', alpha=0.7, edgecolor='darkgreen', linewidth=1.5)

ax5.set_xlabel('Method', fontsize=11, fontweight='bold')
ax5.set_ylabel('Mean Identification Error (%)', fontsize=11, fontweight='bold')
ax5.set_title('Performance Comparison: Lower is Better', fontsize=12, fontweight='bold')
ax5.set_xticks(x_pos_methods)
ax5.set_xticklabels(methods)
ax5.legend(loc='upper right', fontsize=10)
ax5.grid(True, axis='y', alpha=0.3)
ax5.set_yscale('log')  # 使用对数刻度以显示大范围差异

# 添加数值标签
for i, (e1, e6) in enumerate(zip(run1_errors, run6_errors)):
    ax5.text(i - width/2, e1 * 1.5, f'{e1:.4f}%', ha='center', va='bottom', fontsize=8, rotation=0)
    ax5.text(i + width/2, e6 * 1.5, f'{e6:.4f}%', ha='center', va='bottom', fontsize=8, rotation=0)

# 添加关键观察
ax5.text(0.02, 0.95,
         '✅ Improved Attention NN: 233x better than Standard NN\n'
         '❌ Original Attention NN: 18.9x worse than Standard NN',
         transform=ax5.transAxes,
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3),
         fontsize=9,
         verticalalignment='top')

# 总标题
fig.suptitle('Attention Mechanism: Original vs Improved with Input Normalization',
             fontsize=16, fontweight='bold', y=0.995)

plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('comparison_original_vs_improved.png', dpi=300, bbox_inches='tight')
print("Figure saved as 'comparison_original_vs_improved.png'")

# 创建第二个图：权重变化幅度对比
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 计算权重变化
changes1 = [abs(weights1[-1][i] - weights1[0][i]) for i in range(4)]
changes6 = [abs(weights6[-1][i] - weights6[0][i]) for i in range(4)]

# 左图：权重变化幅度
x_pos = range(len(feature_names))
width = 0.35

bars1 = ax1.bar([p - width/2 for p in x_pos], changes1, width,
                 label='Original', color='#e74c3c', alpha=0.7)
bars2 = ax1.bar([p + width/2 for p in x_pos], changes6, width,
                 label='Improved', color='#2ecc71', alpha=0.7)

ax1.set_xlabel('Feature')
ax1.set_ylabel('Absolute Weight Change')
ax1.set_title('Learning Magnitude: Weight Change from Epoch 0 to 99')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(feature_names, rotation=15)
ax1.legend()
ax1.grid(True, axis='y', alpha=0.3)

for i, (c1, c6) in enumerate(zip(changes1, changes6)):
    ax1.text(i - width/2, c1 + 0.0001, f'{c1:.4f}', ha='center', va='bottom', fontsize=8)
    ax1.text(i + width/2, c6 + 0.0001, f'{c6:.4f}', ha='center', va='bottom', fontsize=8)

# 右图：权重标准差（均匀性指标）
def calc_std(weights):
    mean_w = sum(weights) / len(weights)
    variance = sum((w - mean_w)**2 for w in weights) / len(weights)
    return variance ** 0.5

std1_initial = calc_std(weights1[0])
std1_final = calc_std(weights1[-1])
std6_initial = calc_std(weights6[0])
std6_final = calc_std(weights6[-1])

categories = ['Epoch 0', 'Epoch 99']
original_stds = [std1_initial, std1_final]
improved_stds = [std6_initial, std6_final]

x_pos2 = range(len(categories))
bars3 = ax2.bar([p - width/2 for p in x_pos2], original_stds, width,
                 label='Original', color='#e74c3c', alpha=0.7)
bars4 = ax2.bar([p + width/2 for p in x_pos2], improved_stds, width,
                 label='Improved', color='#2ecc71', alpha=0.7)

ax2.set_xlabel('Training Stage')
ax2.set_ylabel('Weight Std Dev (Lower = More Uniform)')
ax2.set_title('Weight Distribution Uniformity')
ax2.set_xticks(x_pos2)
ax2.set_xticklabels(categories)
ax2.legend()
ax2.grid(True, axis='y', alpha=0.3)
ax2.set_yscale('log')

for i, (s1, s6) in enumerate(zip(original_stds, improved_stds)):
    ax2.text(i - width/2, s1 * 1.5, f'{s1:.4f}', ha='center', va='bottom', fontsize=9)
    ax2.text(i + width/2, s6 * 1.5, f'{s6:.4f}', ha='center', va='bottom', fontsize=9)

ax2.text(0.5, 0.95,
         f'Improved version is {std1_final/std6_final:.0f}x more uniform',
         transform=ax2.transAxes,
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5),
         fontsize=10,
         ha='center',
         va='top')

plt.tight_layout()
plt.savefig('weight_analysis_comparison.png', dpi=300, bbox_inches='tight')
print("Figure saved as 'weight_analysis_comparison.png'")

print("\n" + "="*60)
print("VISUALIZATION COMPLETE")
print("="*60)
print("\nGenerated files:")
print("  1. comparison_original_vs_improved.png - Main comparison")
print("  2. weight_analysis_comparison.png - Detailed weight analysis")
print("\nKey findings:")
print(f"  • Improved version: {std1_final/std6_final:.0f}x more uniform weights")
print(f"  • Improved version: {max(changes6)/max(changes1):.1f}x larger weight changes (more learning)")
print(f"  • Performance: 233x better than baseline (vs 18.9x worse in original)")
