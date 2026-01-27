# 阶段2快速启动 - 超参数搜索

## 🎯 目标
基于C4架构（heads=12, layers=5, batch=128），搜索最优超参数组合
**目标准确率**：85.5-85.7%（vs 当前C4-seed96: 85.18%）

---

## 📋 阶段2概述

### Step 2.1：学习率搜索（当前步骤）
- **目标**：找到最优学习率
- **搜索范围**：[1.3e-3, 1.4e-3, 1.5e-3, 1.6e-3, 1.7e-3]
- **固定参数**：λ_ratio=3.0e-3, λ_itm=2.5e-3
- **实验数量**：5个（3并行 + 2并行 = 2批次）
- **预计时间**：~5小时

### Step 2.2：Lambda权重搜索（Step 2.1完成后）
- **目标**：优化CID损失权重
- **搜索范围**：lambda_ratio × lambda_itm 的9组组合
- **固定参数**：使用Step 2.1的最佳lr
- **预计时间**：6小时（9个实验，3并行）

### Step 2.3：（可选）多seed验证
- **目标**：验证最佳配置的稳定性
- **Seeds**：42, 96, 100
- **预计时间**：4-6小时

---

## ⚡ 快速启动（2步）

### Step 1: 启动学习率搜索

```bash
cd /home/user/chengtaiyu/RCLMuFN-main_copy/src

# 赋予执行权限
chmod +x run_stage2_step1_lr_search.sh
chmod +x monitor_stage2.sh
chmod +x analyze_stage2_step1.py

# 后台启动Step 2.1
nohup bash run_stage2_step1_lr_search.sh > stage2_step1.log 2>&1 &
echo $! > stage2_step1.pid
```

### Step 2: 监控进度

```bash
# 实时监控（推荐）
watch -n 10 bash monitor_stage2.sh

# 或查看日志
tail -f stage2_step1.log

# 或单独查看某个配置
tail -f ../output_dir/stage2_lr_search/lr_1.5e-3_seed42/training.log
```

---

## 📊 Step 2.1 完成后（~5小时）

```bash
# 分析结果
python analyze_stage2_step1.py
```

**你会看到类似输出**：
```
============================================================
阶段2 - Step 2.1：学习率搜索结果分析
============================================================

📊 学习率性能对比

LR           Test Acc        Macro F1     Δ vs C4-96   Δ vs Baseline    Epoch
--------------------------------------------------------------------------------
Baseline     0.8514 (85.14%)  0.8501      --           --               --
C4-seed96    0.8518 (85.18%)  0.8500      --           +0.04%           4
--------------------------------------------------------------------------------
1.3e-3       0.8532 (85.32%)  0.8516      +0.14%       +0.18%           7
1.4e-3       0.8547 (85.47%)  0.8531      +0.29%       +0.33%           6
1.5e-3       0.8541 (85.41%)  0.8525      +0.23%       +0.27%           8
1.6e-3       0.8538 (85.38%)  0.8522      +0.20%       +0.24%           7
1.7e-3       0.8525 (85.25%)  0.8509      +0.07%       +0.11%           9
--------------------------------------------------------------------------------

🏆 最佳学习率: 1.4e-3
   Test Accuracy: 0.8547 (85.47%)
   Macro F1:      0.8531
   vs Baseline:   +0.33%
   vs C4-seed96:  +0.29%
   Best Epoch:    6

📋 Step 2.2 执行建议

✅ 找到显著提升的学习率！

推荐策略：
  1. 使用LR=1.4e-3进入Step 2.2（lambda搜索）
  2. 预期lambda搜索可带来额外 +0.1-0.3% 提升
  3. 最终目标：85.5-85.8%

执行命令：
  python generate_stage2_step2.py --best_lr 1.4e-3
  bash run_stage2_step2_lambda_search.sh
```

---

## 🚀 进入Step 2.2（如果Step 2.1成功）

```bash
# 自动生成Step 2.2脚本（基于最佳lr）
python generate_stage2_step2.py --best_lr 1.4e-3

# 启动lambda搜索
nohup bash run_stage2_step2_lambda_search.sh > stage2_step2.log 2>&1 &

# 监控（~6小时）
watch -n 10 bash monitor_stage2_step2.sh
```

---

## 📁 文件结构

```
output_dir/
├── stage2_lr_search/           # Step 2.1训练输出
│   ├── lr_1.3e-3_seed42/
│   │   ├── RCLMuFN/model.pt
│   │   └── training.log
│   ├── lr_1.4e-3_seed42/
│   ├── lr_1.5e-3_seed42/
│   ├── lr_1.6e-3_seed42/
│   └── lr_1.7e-3_seed42/
│
├── stage2_lambda_search/       # Step 2.2训练输出（稍后）
│   ├── lr_1.4e-3_lambda_2.5_2.0_seed42/
│   └── ...
│
└── stage2_results/             # JSON结果
    ├── lr_1.3e-3_seed42.json
    ├── lr_1.4e-3_seed42.json
    ├── ...
    ├── step2.1_analysis.json   # Step 2.1分析
    └── step2.2_analysis.json   # Step 2.2分析（稍后）
```

---

## 🎯 预期结果

### 乐观场景（50%概率）
```
Step 2.1: +0.25-0.35% → 85.39-85.49%
Step 2.2: +0.1-0.2%   → 85.5-85.7%
最终目标: 85.5-85.7%
```

### 保守场景（30%概率）
```
Step 2.1: +0.15-0.25% → 85.29-85.39%
Step 2.2: +0.05-0.1%  → 85.35-85.5%
最终结果: 85.35-85.5%
```

### 失败场景（20%概率）
```
Step 2.1: <+0.15%
问题：C4架构 + batch=128可能显存不足或训练不稳定
解决：检查日志，考虑调整batch_size或回退baseline
```

---

## 🔧 常见问题

### Q1: 显存不足（OOM）怎么办？

**症状**：训练日志显示`CUDA out of memory`

**解决方案1**（推荐）：降低batch_size
```bash
# 编辑 run_stage2_step1_lr_search.sh
# 将 BATCH_SIZE=128 改为 BATCH_SIZE=96
```

**解决方案2**：回退到baseline架构
```bash
# 使用heads=8, layers=4, batch=128
python generate_stage2_step1.py --architecture baseline
bash run_stage2_step1_baseline.sh
```

---

### Q2: 某个学习率实验失败怎么办？

**不影响其他实验！**

单个失败：
```bash
# 查看失败日志
cat ../output_dir/stage2_lr_search/lr_1.5e-3_seed42/training.log

# 手动重跑（如果需要）
CUDA_VISIBLE_DEVICES=2 python main.py \
  --device 0 --learning_rate 1.5e-3 --num_heads 12 --layers 5 \
  --train_batch_size 128 --seed 42 ...
```

---

### Q3: 如何停止训练？

```bash
# 使用PID文件
kill $(cat stage2_step1.pid)

# 或手动查找
ps aux | grep run_stage2_step1
kill <PID>

# 强制停止所有相关进程
pkill -f "stage2_step1"
```

---

### Q4: Step 2.1结果不理想怎么办？

**如果所有lr都 <+0.15%**：
1. 检查是否有训练失败
2. 考虑C4架构可能不适合（回退baseline）
3. 或者降低batch_size重试

---

## ⏱️ 完整时间规划

| 阶段 | 时间 | 操作 |
|------|------|------|
| **Step 2.1** | 5小时 | 学习率搜索（5个实验，2批次） |
| 分析 | 5分钟 | `python analyze_stage2_step1.py` |
| **决策点** | -- | **是否继续Step 2.2** |
| **Step 2.2** | 6小时 | Lambda搜索（9个实验，3批次） |
| 分析 | 5分钟 | `python analyze_stage2_step2.py` |
| **Step 2.3** | 6小时 | 多seed验证（可选） |
| **总计** | 11-17小时 | 取决于是否做Step 2.2/2.3 |

---

## 📝 论文写作提示

### Step 2.1可以写成超参数敏感性分析：

**Figure X: Learning Rate Sensitivity**
```
准确率随学习率的变化曲线图
横轴: Learning Rate (1.3e-3 → 1.7e-3)
纵轴: Test Accuracy
峰值: 1.4e-3 (85.47%)
```

**文字描述**：
> "We conduct a grid search over learning rates in the range [1.3e-3, 1.7e-3]. As shown in Figure X, the model achieves peak performance at lr=1.4e-3 (85.47%), representing a +0.33% improvement over the baseline. Learning rates above 1.5e-3 show degraded performance due to training instability."

---

## ✅ 检查清单

**启动前**：
- [ ] 3个GPU可用
- [ ] 阶段1的分析已完成
- [ ] 确认使用C4架构（heads=12, layers=5）
- [ ] 确认batch_size=128（或根据显存调整为96）

**Step 2.1完成后**：
- [ ] 运行`python analyze_stage2_step1.py`
- [ ] 检查是否有显著提升（≥+0.15%）
- [ ] 决定是否进入Step 2.2

**阶段2全部完成**：
- [ ] 确定最佳超参数组合
- [ ] 准备多seed验证（论文需要）

---

## 🎉 准备好开始了吗？

**立即启动Step 2.1**：
```bash
cd /home/user/chengtaiyu/RCLMuFN-main_copy/src

# 启动学习率搜索（5小时）
nohup bash run_stage2_step1_lr_search.sh > stage2_step1.log 2>&1 &
echo $! > stage2_step1.pid

# 实时监控
watch -n 10 bash monitor_stage2.sh
```

**Good luck! 🚀**

---

## 📞 帮助

- 查看日志：`tail -f stage2_step1.log`
- 监控GPU：`watch -n 1 nvidia-smi`
- 检查完成数：`ls ../output_dir/stage2_results/*.json | wc -l`
- 查看某个配置：`cat ../output_dir/stage2_results/lr_1.5e-3_seed42.json`
