# 重新运行实验指南

## 问题分析
当前注意力机制存在的问题：
- **训练从第一个epoch就将velocity权重固定在~0.999**
- 原因：输入特征尺度差异导致attention立即饱和（velocity: 10-30, 其他特征: ±0.1）
- 缺少真正的学习过程，虽然结果物理上合理但是"侥幸"而非"学习"

## 改进方案
已在 `experiment.py` 中实现了以下改进：

### 1. 输入特征标准化 (BatchNorm)
```python
self.input_norm = nn.BatchNorm1d(input_size, affine=True, track_running_stats=True)
```
- 将所有输入特征标准化到相同尺度
- 防止attention机制被特征尺度差异主导

### 2. 小初始化权重
```python
nn.init.uniform_(self.attention_logits.weight, -0.01, 0.01)
nn.init.zeros_(self.attention_logits.bias)
```
- 让初始attention权重更均匀分布
- 给模型探索不同特征组合的机会

### 3. 可学习的温度参数
```python
self.temperature = nn.Parameter(torch.ones(1) * 5.0)
```
- 控制attention分布的锐度
- 初始温度较高，允许更平滑的权重分布

## 运行实验的三种方案

### 方案A：快速对比测试（推荐）
测试改进后的attention机制是否解决了饱和问题

```bash
cd D:\03_Proj\00_GitProj\AI-Scientist\results\steering_parameter_identification\20260111_220815_attention_parameter_identification

# 创建新的运行目录
mkdir run_6

# 运行改进版本（默认使用input_norm=True）
python experiment.py --out_dir run_6 --noise_level 0.02 --seed 123
```

**预期结果**：
- attention权重应该在训练过程中逐渐演变
- 第一个epoch权重应该相对均匀（例如 [0.2, 0.3, 0.25, 0.25]）
- 最终仍然收敛到velocity占主导，但经历了学习过程

---

### 方案B：完整对比实验（4个noise level × 2个版本）
对比有无输入标准化的效果

#### B1. 使用输入标准化版本（改进版）

```bash
# noise_level = 0.01
python experiment.py --out_dir run_6_norm_001 --noise_level 0.01 --seed 100

# noise_level = 0.02
python experiment.py --out_dir run_7_norm_002 --noise_level 0.02 --seed 101

# noise_level = 0.05
python experiment.py --out_dir run_8_norm_005 --noise_level 0.05 --seed 102

# noise_level = 0.1
python experiment.py --out_dir run_9_norm_010 --noise_level 0.1 --seed 103
```

#### B2. 禁用输入标准化（复现原问题）
修改 experiment.py 第543行，将 `use_input_norm=True` 改为 `False`：

```python
# 找到这一行（约第543行）：
attention_model = AttentionParameterIdentificationNet(hidden_sizes, activation)

# 改为：
attention_model = AttentionParameterIdentificationNet(hidden_sizes, activation, use_input_norm=False)
```

然后运行：
```bash
# noise_level = 0.02 (无标准化)
python experiment.py --out_dir run_10_no_norm_002 --noise_level 0.02 --seed 101
```

---

### 方案C：创建对比脚本（最系统）
创建一个批量运行脚本

#### 1. 创建批处理脚本 `run_comparison.sh` (Linux/Mac) 或 `run_comparison.bat` (Windows)

**Windows版本** (`run_comparison.bat`):
```batch
@echo off
cd D:\03_Proj\00_GitProj\AI-Scientist\results\steering_parameter_identification\20260111_220815_attention_parameter_identification

echo ========================================
echo Running IMPROVED version with input normalization
echo ========================================

python experiment.py --out_dir run_6_improved_001 --noise_level 0.01 --seed 100
python experiment.py --out_dir run_7_improved_002 --noise_level 0.02 --seed 101
python experiment.py --out_dir run_8_improved_005 --noise_level 0.05 --seed 102
python experiment.py --out_dir run_9_improved_010 --noise_level 0.1 --seed 103

echo ========================================
echo Experiments completed!
echo ========================================
```

**Linux/Mac版本** (`run_comparison.sh`):
```bash
#!/bin/bash
cd /path/to/AI-Scientist/results/steering_parameter_identification/20260111_220815_attention_parameter_identification

echo "========================================"
echo "Running IMPROVED version with input normalization"
echo "========================================"

python experiment.py --out_dir run_6_improved_001 --noise_level 0.01 --seed 100
python experiment.py --out_dir run_7_improved_002 --noise_level 0.02 --seed 101
python experiment.py --out_dir run_8_improved_005 --noise_level 0.05 --seed 102
python experiment.py --out_dir run_9_improved_010 --noise_level 0.1 --seed 103

echo "========================================"
echo "Experiments completed!"
echo "========================================"
```

#### 2. 运行脚本
```bash
# Windows
run_comparison.bat

# Linux/Mac
chmod +x run_comparison.sh
./run_comparison.sh
```

---

## 在服务器上运行（重要）

### 1. 同步代码到服务器
```bash
# 假设你的服务器地址是 user@server.com
# 从本地同步修改后的 experiment.py 到服务器

# 方法1：使用 rsync
rsync -avz D:\03_Proj\00_GitProj\AI-Scientist\results\steering_parameter_identification\20260111_220815_attention_parameter_identification\ user@server.com:/path/to/AI-Scientist/results/steering_parameter_identification/20260111_220815_attention_parameter_identification/

# 方法2：使用 scp
scp experiment.py user@server.com:/path/to/AI-Scientist/results/steering_parameter_identification/20260111_220815_attention_parameter_identification/
```

### 2. SSH登录服务器
```bash
ssh user@server.com
```

### 3. 激活环境并运行
```bash
# 进入实验目录
cd /path/to/AI-Scientist/results/steering_parameter_identification/20260111_220815_attention_parameter_identification

# 激活conda环境
conda activate ai_scientist

# 确认PyTorch可用
python -c "import torch; print(torch.__version__)"

# 运行实验（推荐使用nohup或tmux以防断连）
nohup python experiment.py --out_dir run_6 --noise_level 0.02 --seed 123 > run_6.log 2>&1 &

# 或使用tmux（推荐）
tmux new -s experiment
python experiment.py --out_dir run_6 --noise_level 0.02 --seed 123
# 按 Ctrl+B 然后 D 来detach
```

### 4. 监控运行状态
```bash
# 查看日志
tail -f run_6.log

# 或者查看进程
ps aux | grep experiment.py

# 使用GPU的话，监控GPU使用
watch -n 1 nvidia-smi
```

### 5. 运行完成后同步结果回本地
```bash
# 从服务器同步结果到本地
rsync -avz user@server.com:/path/to/AI-Scientist/results/steering_parameter_identification/20260111_220815_attention_parameter_identification/run_6/ D:\03_Proj\00_GitProj\AI-Scientist\results\steering_parameter_identification\20260111_220815_attention_parameter_identification\run_6\
```

---

## 查看和分析结果

### 1. 检查attention权重演变
```bash
cd D:\03_Proj\00_GitProj\AI-Scientist\results\steering_parameter_identification\20260111_220815_attention_parameter_identification

# 查看第一个epoch和最后一个epoch的权重
python -c "
import json
data = json.load(open('run_6/results.json'))
weights_hist = data['attention_nn']['attention_weights']
print('Epoch 0 weights:', weights_hist[0])
print('Epoch 99 weights:', weights_hist[-1])
print()
print('Features: [delta, velocity, beta, yaw_rate]')
"
```

**判断改进是否成功**：
- ✅ **成功**：Epoch 0 权重相对均匀（例如 [0.2, 0.3, 0.25, 0.25]），然后逐渐演变
- ❌ **失败**：Epoch 0 权重就已经是 [0.0, 0.999, 0.0, 0.0]，和原来一样

### 2. 绘制attention权重演变曲线
```bash
python plot.py
```

这会生成或更新以下图片：
- `attention_weights.png` - 每个特征的权重随训练演变
- `final_attention_weights.png` - 最终权重对比
- `error_comparison.png` - 不同方法的误差对比

### 3. 查看数值结果
```bash
# 查看最终误差
python -c "
import json
for run_id in [1, 6]:  # 对比原版(run_1)和改进版(run_6)
    try:
        data = json.load(open(f'run_{run_id}/results.json'))
        attn_error = data['attention_nn']['mean_error_percent']
        print(f'Run {run_id}: Attention NN mean error = {attn_error:.6f}%')
    except FileNotFoundError:
        print(f'Run {run_id}: Not found')
"
```

### 4. 对比温度参数的演变
```bash
# 查看temperature参数的最终值（如果保存了的话）
python -c "
import torch
import json
# 如果model checkpoint被保存，可以加载查看temperature
# 否则只能从training log中分析
data = json.load(open('run_6/results.json'))
print('Final attention weights:', data['attention_nn']['final_attention_weights'])
"
```

---

## 预期结果对比

### 原版本（有问题的）
```
Epoch 0: [0.0006, 0.9992, 0.0002, 0.00001]  # 从一开始就饱和
Epoch 99: [0.00003, 0.9999, 0.000008, 0.0000005]
```

### 改进版本（预期）
```
Epoch 0: [0.22, 0.31, 0.24, 0.23]  # 相对均匀
Epoch 10: [0.15, 0.52, 0.18, 0.15]  # 逐渐演变
Epoch 50: [0.05, 0.82, 0.08, 0.05]
Epoch 99: [0.02, 0.93, 0.03, 0.02]  # 最终收敛，但经历了学习过程
```

---

## 故障排除

### 问题1：BatchNorm报错
如果batch_size=1导致BatchNorm报错：
```python
# 修改 experiment.py 第543行附近
attention_model = AttentionParameterIdentificationNet(hidden_sizes, activation, use_input_norm=False)
```

### 问题2：找不到模块
```bash
# 确认环境
conda activate ai_scientist
pip install torch torchvision scipy numpy
```

### 问题3：GPU内存不足
```bash
# 减小batch_size
python experiment.py --out_dir run_6 --noise_level 0.02 --batch_size 32
```

### 问题4：想恢复原版本
```bash
# 从run_1.py复制原始版本
cp run_1.py experiment.py
```

---

## 进一步分析建议

### 1. 可视化attention权重演变动画
创建 `plot_attention_evolution.py`：
```python
import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

data = json.load(open('run_6/results.json'))
weights_hist = data['attention_nn']['attention_weights']

fig, ax = plt.subplots(figsize=(10, 6))
feature_names = ['delta', 'velocity', 'beta', 'yaw_rate']
colors = ['blue', 'red', 'green', 'orange']

def animate(epoch):
    ax.clear()
    weights = weights_hist[epoch]
    ax.bar(feature_names, weights, color=colors)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Attention Weight')
    ax.set_title(f'Epoch {epoch}')
    ax.grid(True, alpha=0.3)

anim = animation.FuncAnimation(fig, animate, frames=len(weights_hist), interval=50)
anim.save('attention_evolution.gif', writer='pillow', fps=20)
print("Animation saved to attention_evolution.gif")
```

### 2. 统计分析
```python
import json
import numpy as np

data = json.load(open('run_6/results.json'))
weights_hist = np.array(data['attention_nn']['attention_weights'])

print("Weight statistics:")
print("="*50)
for i, name in enumerate(['delta', 'velocity', 'beta', 'yaw_rate']):
    feature_weights = weights_hist[:, i]
    print(f"\n{name}:")
    print(f"  Initial: {feature_weights[0]:.4f}")
    print(f"  Final: {feature_weights[-1]:.4f}")
    print(f"  Mean: {feature_weights.mean():.4f}")
    print(f"  Std: {feature_weights.std():.4f}")
    print(f"  Change: {feature_weights[-1] - feature_weights[0]:.4f}")
```

---

## 总结

改进后的attention机制应该展现出：
1. ✅ 初始权重更均匀（~0.25 each）
2. ✅ 训练过程中逐渐学习特征重要性
3. ✅ 最终仍然收敛到velocity占主导（物理合理）
4. ✅ 但这次是通过学习得到的，而非初始化偏差

如果改进成功，可以在论文中添加：
- "通过输入标准化和温度调节，attention机制展现出清晰的学习轨迹"
- "初始权重均匀分布，经过训练逐渐收敛到物理上合理的特征分布"
- 对比图展示有无改进的attention演变差异
