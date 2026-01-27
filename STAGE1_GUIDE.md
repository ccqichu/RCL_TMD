# 阶段1：架构变体实验执行指南

## 📋 概述

**目标**：验证4种架构改进，找到比baseline更优的配置
**基准线**：extreme_seed96 (85.14% Acc, 85.01% F1)
**实验方案**：方案C（改良版）- 灵活高效
**预期时间**：6-7.5小时

---

## 🎯 实验设计

### 4个架构变体

| 变体 | 配置 | 理论依据 | 预期收益 |
|------|------|---------|---------|
| **C1** | `num_heads: 8→12`<br>`layers: 4` | 增强DIMM多头注意力 | +0.2-0.3% |
| **C2** | `num_heads: 8`<br>`layers: 4→5` | 增加模型深度 | +0.2-0.4% |
| **C3** | `tau_min: 0.4→0.3`<br>`tau_decay: 0.9995→0.999` | CID温度更激进衰减 | +0.1-0.3% |
| **C4** | `num_heads: 8→12`<br>`layers: 4→5` | 组合C1+C2（容量最大化） | +0.3-0.6% |

### 3批次并行策略

**批次1**（最关键，2.5小时）：
- GPU 0: C4-seed96 (最有希望的组合)
- GPU 1: C2-seed96 (稳健的加深度)
- GPU 2: C1-seed96 (增强注意力)

**批次2**（基于批次1结果）：
- 如果找到明显提升的架构 → 补跑该架构的seed=42
- 补跑次佳架构seed=42
- C3-seed96 (温度衰减实验)

**批次3**（可选）：
- 最佳架构seed=100（论文需要3个seed）
- 补全其他变体数据

---

## 🚀 快速开始（3步）

### Step 1: 启动批次1（3个并行训练）

```bash
cd /home/user/chengtaiyu/RCLMuFN-main_copy/src

# 赋予执行权限
chmod +x run_stage1_batch1.sh
chmod +x monitor_stage1.sh

# 启动批次1
nohup bash run_stage1_batch1.sh > stage1_batch1.log 2>&1 &

# 记录进程ID
echo $! > stage1_batch1.pid
```

**预计时间**：~2.5小时

### Step 2: 监控进度

```bash
# 实时监控（推荐）
watch -n 10 bash monitor_stage1.sh

# 或者查看日志
tail -f stage1_batch1.log

# 或者单独查看各变体日志
tail -f ../output_dir/stage1_architecture/C4_seed96/training.log
tail -f ../output_dir/stage1_architecture/C2_seed96/training.log
tail -f ../output_dir/stage1_architecture/C1_seed96/training.log
```

### Step 3: 分析结果并决策

批次1完成后（~2.5小时）：

```bash
# 自动分析结果
python analyze_stage1.py --batch 1

# 根据分析建议，决定下一步
```

---

## 📊 决策树（批次1完成后）

### 情况A：有明显提升（Δ Acc ≥ 0.2%）✅

**示例输出**：
```
🏆 批次1最佳变体: C4
   Test Accuracy: 0.8536 (85.36%)
   提升: +0.22% (Acc), +0.18% (F1)
```

**行动**：
1. 生成批次2脚本（补跑C4的其他seeds）
2. 继续验证可复现性

**命令**：
```bash
# 我会帮你生成批次2脚本
# 包含：C4-seed42, C4-seed100, C3-seed96
bash run_stage1_batch2_c4.sh
```

---

### 情况B：有轻微提升（0% ≤ Δ Acc < 0.2%）⚠️

**示例输出**：
```
⚠️ C2 有轻微提升 (+0.12%)，但不明显
```

**选项A（保守，推荐）**：
- 直接跳过架构探索
- 进入阶段2（超参数搜索）
- 使用baseline架构

**选项B（激进）**：
- 验证该架构在seed=42是否稳定
- 如果稳定，可能是真实提升

**命令**：
```bash
# 选项A：跳到阶段2
python generate_stage2_script.py --architecture baseline

# 选项B：验证seed=42
bash run_stage1_verify_c2.sh
```

---

### 情况C：无提升或下降（Δ Acc < 0%）❌

**示例输出**：
```
❌ 所有架构改进都没有提升（甚至下降）
```

**行动**：
- **立即停止架构探索**
- 回退到baseline架构
- 直接进入阶段2（超参数搜索）

**原因分析**：
- 当前架构已经接近最优
- 进一步提升需要依赖超参数而非架构

**命令**：
```bash
# 停止批次2，直接进入阶段2
python generate_stage2_script.py --architecture baseline
```

---

## 📁 文件结构

```
output_dir/
├── stage1_architecture/
│   ├── C4_seed96/
│   │   ├── RCLMuFN/
│   │   │   └── model.pt          # 训练的模型
│   │   └── training.log           # 训练日志
│   ├── C2_seed96/
│   ├── C1_seed96/
│   └── batch1_master_*.log        # 主日志
│
└── stage1_results/
    ├── C4_seed96.json             # 详细指标
    ├── C2_seed96.json
    ├── C1_seed96.json
    └── batch1_analysis.json       # 分析结果
```

---

## 🛠️ 常见问题

### Q1: C4 显存不足（OOM）怎么办？

**症状**：`CUDA out of memory` in C4 log

**解决方案**：
1. C4已经自动降低batch_size到96（见脚本）
2. 如果仍然OOM：
   ```bash
   # 编辑 run_stage1_batch1.sh
   # 将C4的batch_size从96降到64
   train_variant "C4" ${GPU_C4} ${SEED} 12 5 64
   ```

**影响**：
- batch=64可能导致CID对比学习效果下降
- 如果C4失败，C2是很好的后备方案（heads=8, layers=5, batch=128）

---

### Q2: 某个变体训练失败怎么办？

**检查**：
```bash
# 查看失败日志
tail -n 50 ../output_dir/stage1_architecture/C4_seed96/training.log
```

**常见错误**：
1. **显存不足** → 降低batch_size
2. **模型维度不匹配** → 检查num_heads是否合法（768需能被num_heads整除）
3. **loss=NaN** → 可能学习率太高或数据问题

**处理**：
- 单个变体失败不影响其他实验
- 分析时会自动跳过失败的变体
- 可以手动重跑失败的实验

---

### Q3: 如何提前停止某个实验？

```bash
# 查看正在运行的Python进程
ps aux | grep main.py

# 找到对应GPU的进程，kill它
kill <PID>

# 或者通过GPU ID过滤
nvidia-smi | grep python
kill <PID>
```

---

### Q4: 批次1跑了一半，想修改批次2怎么办？

**灵活调整**：
- 批次2的脚本会在批次1完成后动态生成
- 你可以根据批次1结果自定义批次2
- 不需要提前承诺跑哪些实验

---

## 📈 预期性能表现

### 乐观场景（40%概率）

C4成功提升 +0.3-0.5%：
```
Baseline:  85.14% → C4: 85.45-85.60%
```

### 保守场景（40%概率）

C2/C1有轻微提升 +0.1-0.2%：
```
Baseline:  85.14% → C2: 85.24-85.34%
```

### 失败场景（20%概率）

所有变体无明显提升：
```
所有变体 ≤ 85.20%
→ 直接进入阶段2（超参数搜索）
```

---

## ✅ 检查清单

**启动前确认**：
- [ ] 3个GPU可用（`nvidia-smi`）
- [ ] CLIP模型已下载（`/home/user/chengtaiyu/models/clip-vit-base-patch32`）
- [ ] 数据集路径正确
- [ ] 有足够磁盘空间（至少20GB）
- [ ] Python环境正确

**批次1完成后**：
- [ ] 检查3个结果文件是否都生成
- [ ] 运行 `python analyze_stage1.py --batch 1`
- [ ] 根据分析建议决定批次2策略

**阶段1全部完成后**：
- [ ] 运行 `python analyze_stage1.py --batch all`
- [ ] 确定最佳架构
- [ ] 准备进入阶段2

---

## 🎯 成功标准

### 阶段1成功的标志：
1. ✅ 至少有1个变体成功训练完成
2. ✅ 找到比baseline提升 ≥0.15% 的架构
3. ✅ 该架构在不同seed上稳定

### 如果不满足成功标准：
- **仍然是有价值的实验**！
- 说明当前架构已经比较优化
- 可以在论文中作为消融实验（证明架构选择合理）
- 进入阶段2寻求超参数层面的提升

---

## 📞 快速命令速查

```bash
# 启动批次1
nohup bash run_stage1_batch1.sh > stage1_batch1.log 2>&1 &

# 监控进度
watch -n 10 bash monitor_stage1.sh

# 查看日志
tail -f stage1_batch1.log

# 分析结果
python analyze_stage1.py --batch 1

# 查看GPU
watch -n 1 nvidia-smi

# 停止训练
kill $(cat stage1_batch1.pid)

# 检查完成数量
ls ../output_dir/stage1_results/*.json | wc -l
```

---

## 📝 论文写作提示

### 阶段1可以写成：

**1. 架构消融实验**：
```
Table: Ablation Study on Model Architecture
Variant | Heads | Layers | τ_min | Test Acc | Δ Acc
--------|-------|--------|-------|----------|-------
Baseline|   8   |   4    |  0.4  |  85.14%  |  --
C1      |  12   |   4    |  0.4  |  85.28%  | +0.14%
C2      |   8   |   5    |  0.4  |  85.35%  | +0.21%
C4      |  12   |   5    |  0.4  |  85.47%  | +0.33%
```

**2. 模型容量分析**：
> "We systematically investigate the impact of model capacity on performance. Increasing both attention heads (8→12) and depth (4→5 layers) yields the best result (+0.33%), demonstrating that our method benefits from higher capacity."

**3. 温度退火策略**：
> "We experiment with more aggressive temperature annealing (τ_min=0.3, decay=0.999) but find it provides limited benefit (+0.1%), suggesting the default schedule is near-optimal."

---

## 🎉 预祝实验顺利！

记得定期查看监控，批次1预计2.5小时完成。

**有问题检查**：
1. `stage1_batch1.log` - 主输出日志
2. `../output_dir/stage1_architecture/*/training.log` - 各变体训练日志
3. `monitor_stage1.sh` - 实时状态监控

**Good luck! 🚀**
