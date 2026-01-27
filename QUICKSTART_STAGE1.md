# 阶段1快速启动 - 3步开始实验

## 🎯 目标
验证4种架构改进，找到比extreme_seed96 (85.14%)更优的配置

## ⚡ 3步启动

### Step 1: 环境测试（5分钟）

```bash
cd /home/user/chengtaiyu/RCLMuFN-main_copy/src

# 测试环境配置
chmod +x test_stage1_setup.sh
bash test_stage1_setup.sh
```

**如果看到 `✅ 所有测试通过！`** → 进入Step 2
**如果失败** → 查看错误日志，修复后重试

---

### Step 2: 启动批次1（3个并行，~2.5小时）

```bash
# 赋予执行权限
chmod +x run_stage1_batch1.sh
chmod +x monitor_stage1.sh

# 后台启动批次1
nohup bash run_stage1_batch1.sh > stage1_batch1.log 2>&1 &

# 记录进程ID
echo $! > stage1_batch1.pid
```

**批次1将运行**：
- GPU 0: C4 (heads=12, layers=5) - 最大容量
- GPU 1: C2 (heads=8, layers=5) - 加深度
- GPU 2: C1 (heads=12, layers=4) - 增强注意力

---

### Step 3: 监控&分析

#### 实时监控
```bash
# 方式1：实时监控面板（推荐）
watch -n 10 bash monitor_stage1.sh

# 方式2：查看主日志
tail -f stage1_batch1.log

# 方式3：查看各变体日志
tail -f ../output_dir/stage1_architecture/C4_seed96/training.log
```

#### 批次1完成后（~2.5小时）

```bash
# 自动分析结果
python analyze_stage1.py --batch 1
```

**你会看到**：
```
🏆 批次1最佳变体: C4
   Test Accuracy: 0.8536 (85.36%)
   提升: +0.22% (Acc), +0.18% (F1)

📋 批次2执行建议
✅ C4 有明显提升！
推荐批次2策略：
  1. C4-seed42  (验证可复现性)
  2. C4-seed100 (扩展验证)
  3. C3-seed96  (补充温度衰减实验)

执行命令：
  python generate_batch2.py --best_variant C4
  bash run_stage1_batch2_c4.sh
```

---

## 📊 决策树

### 如果有明显提升（Δ Acc ≥ 0.2%）✅
```bash
# 生成并执行批次2
python generate_batch2.py --best_variant C4
bash run_stage1_batch2_c4.sh

# 等待批次2完成后
python analyze_stage1.py --batch all
```

### 如果轻微提升（0% ≤ Δ Acc < 0.2%）⚠️
**选项A（推荐）**：直接进入阶段2
```bash
# 跳过架构探索，进入超参数搜索
python generate_stage2_script.py --architecture baseline
```

**选项B**：验证该架构
```bash
python generate_batch2.py --best_variant C2 --strategy verify
bash run_stage1_batch2_c2.sh
```

### 如果无提升（Δ Acc < 0%）❌
```bash
# 立即停止，进入阶段2
python generate_stage2_script.py --architecture baseline
```

---

## 📁 文件清单

**已生成的脚本**：
- ✅ `run_stage1_batch1.sh` - 批次1训练脚本（3并行）
- ✅ `monitor_stage1.sh` - 实时监控脚本
- ✅ `analyze_stage1.py` - 结果分析脚本
- ✅ `generate_batch2.py` - 批次2动态生成器
- ✅ `test_stage1_setup.sh` - 环境测试脚本
- ✅ `STAGE1_GUIDE.md` - 完整执行指南

**运行时生成**：
- `run_stage1_batch2_c4.sh` - 批次2脚本（运行generate_batch2.py后生成）

**输出目录**：
```
output_dir/
├── stage1_architecture/    # 模型和训练日志
│   ├── C4_seed96/
│   ├── C2_seed96/
│   └── C1_seed96/
└── stage1_results/         # JSON结果和分析
    ├── C4_seed96.json
    ├── C2_seed96.json
    ├── C1_seed96.json
    └── batch1_analysis.json
```

---

## 🔧 常见问题

### Q: GPU数量不足3个怎么办？
修改 `run_stage1_batch1.sh` 中的GPU分配：
```bash
# 如果只有2个GPU
GPU_C4=0
GPU_C2=1
GPU_C1=0  # 与C4共用GPU 0（顺序执行）
```

### Q: C4显存不足（OOM）？
脚本已自动将C4的batch_size降到96。如果仍OOM：
```bash
# 编辑 run_stage1_batch1.sh
# 将C4的batch_size从96降到64
train_variant "C4" ${GPU_C4} ${SEED} 12 5 64
```

### Q: 如何停止训练？
```bash
# 方式1：使用PID文件
kill $(cat stage1_batch1.pid)

# 方式2：查找并kill
ps aux | grep run_stage1_batch1.sh
kill <PID>
```

### Q: 某个变体失败了？
**不影响其他实验**！
- 分析时会自动跳过失败的变体
- 查看失败日志：`cat ../output_dir/stage1_architecture/C4_seed96/training.log`
- 可以手动重跑单个变体

---

## ⏱️ 时间规划

| 阶段 | 时间 | 操作 |
|------|------|------|
| 环境测试 | 5分钟 | `bash test_stage1_setup.sh` |
| 批次1 | 2.5小时 | 3个并行训练 |
| 分析结果 | 2分钟 | `python analyze_stage1.py --batch 1` |
| **决策点** | - | **根据结果选择方案** |
| 批次2 (可选) | 2.5小时 | 验证最佳架构 |
| 最终分析 | 2分钟 | `python analyze_stage1.py --batch all` |

**最快完成**：3小时（批次1发现明显提升，直接进阶段2）
**完整验证**：6小时（批次1+批次2）

---

## 📈 预期结果

### 乐观场景（40%概率）
```
C4成功提升 +0.3-0.5%
Baseline 85.14% → C4: 85.45-85.60%
→ 使用C4架构进入阶段2
```

### 保守场景（40%概率）
```
C2/C1轻微提升 +0.1-0.2%
Baseline 85.14% → C2: 85.24-85.34%
→ 选择是否使用C2或回退baseline
```

### 失败场景（20%概率）
```
所有变体 ≤ 85.20%
→ 回退baseline，直接进阶段2
→ 在论文中作为消融实验
```

---

## ✅ 检查清单

**启动前**：
- [ ] 运行环境测试（`test_stage1_setup.sh`）
- [ ] 3个GPU可用（或已调整GPU分配）
- [ ] 磁盘空间 ≥ 20GB
- [ ] 阅读 `STAGE1_GUIDE.md`（可选）

**批次1完成后**：
- [ ] 运行 `analyze_stage1.py --batch 1`
- [ ] 检查是否有明显提升
- [ ] 决定是否运行批次2

**阶段1全部完成**：
- [ ] 运行 `analyze_stage1.py --batch all`
- [ ] 确定进入阶段2的架构
- [ ] 准备超参数搜索

---

## 🎉 准备好了吗？

**立即开始**：
```bash
cd /home/user/chengtaiyu/RCLMuFN-main_copy/src

# Step 1: 测试
bash test_stage1_setup.sh

# Step 2: 启动
nohup bash run_stage1_batch1.sh > stage1_batch1.log 2>&1 &

# Step 3: 监控
watch -n 10 bash monitor_stage1.sh
```

**Good luck! 🚀**

---

## 📞 帮助

- 完整指南：`cat STAGE1_GUIDE.md`
- 查看日志：`tail -f stage1_batch1.log`
- 监控GPU：`watch -n 1 nvidia-smi`
- 检查进度：`ls ../output_dir/stage1_results/*.json | wc -l`
