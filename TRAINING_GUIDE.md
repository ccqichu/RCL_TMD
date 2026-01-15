# MMSD Dataset Training Guide - Single-Stage Mode

## 已应用的优化

### 1. 模型参数修改 (model.py)
- ✅ `rho=0.4` (原 0.3) - Vision 一致性目标增加 33%
- ✅ `rho_t=0.6` (原 0.5) - Text 一致性目标增加 20%

### 2. 创建的训练脚本

| 脚本名称 | 推荐优先级 | 说明 | 关键参数变化 |
|---------|----------|------|------------|
| **train_single_stage.sh** | ⭐⭐⭐ 最推荐 | 平衡的优化配置 | lambda_ratio: 5e-3, lambda_itm: 3e-3, CLIP LR: 5e-6 |
| train_exp_aggressive.sh | ⭐⭐ 激进版 | 更强的CID监督 | lambda_ratio: 1e-2, lambda_itm: 5e-3, CLIP LR: 1e-5, cosine schedule |
| train_exp_conservative.sh | ⭐ 保守版 | 最小风险改动 | lambda_ratio: 3e-3, lambda_itm: 2e-3, CLIP LR: 3e-6 |
| train_exp_hard_neg.sh | ⭐⭐ 困难负样本 | 使用低相似度负样本 | neg_sampling: low_sim |

---

## 快速开始

### 推荐流程：先运行基线优化版本

```bash
# 1. 赋予执行权限
chmod +x train_single_stage.sh

# 2. 运行训练 (推荐!)
bash train_single_stage.sh
```

**预计训练时间:** 约 16-20 小时 (12 epochs, batch_size=24)

**显存需求:** 约 18-22GB (需要确保 GPU 显存充足)

---

## 参数对比表

### 关键超参数变化

| 参数 | 原始 (stage2) | 基线优化 | 激进版 | 保守版 |
|------|-------------|---------|--------|--------|
| **Epochs** | 8 | **12** | 12 | 10 |
| **lambda_ratio_end** | 2e-3 | **5e-3** ↑2.5x | 1e-2 ↑5x | 3e-3 ↑1.5x |
| **lambda_itm_end** | 1.5e-3 | **3e-3** ↑2x | 5e-3 ↑3.3x | 2e-3 ↑1.3x |
| **clip_learning_rate** | 1e-6 | **5e-6** ↑5x | 1e-5 ↑10x | 3e-6 ↑3x |
| **lambda_warmup_epochs** | 2 | **3** | 2 | 2 |
| **lambda_ramp_epochs** | 3 | **5** | 4 | 3 |
| **lambda_schedule** | linear | **linear** | cosine | linear |
| **tau_schedule_mode** | epoch | **epoch** | step | epoch |
| **neg_sampling** | label_aware | **label_aware** | label_aware | label_aware |

### 模型内部参数 (model.py:633-634)

| 参数 | 原始 | 优化后 | 说明 |
|------|-----|--------|------|
| **rho** (vision) | 0.3 | **0.4** | 期望更多 patch 为一致性 |
| **rho_t** (text) | 0.5 | **0.6** | 期望更多 token 为一致性 |

---

## Lambda 权重调度时间表

### 基线优化版 (train_single_stage.sh)

| Epoch | lambda_ratio | lambda_itm | 阶段 |
|-------|--------------|------------|------|
| 0-2   | 0.0000       | 0.0000     | Warmup (无 CID 损失) |
| 3     | 0.0010       | 0.0006     | Ramp 开始 |
| 4     | 0.0020       | 0.0012     | |
| 5     | 0.0030       | 0.0018     | |
| 6     | 0.0040       | 0.0024     | |
| 7     | 0.0050       | 0.0030     | 完全激活 |
| 8-11  | 0.0050       | 0.0030     | 全强度训练 |

---

## 监控指标

### WandB 重点关注

训练过程中需要关注的关键指标：

#### 性能指标 (最重要)
- **macro_test_f1** - 主要优化目标 (MMSD 类别可能不平衡)
- **test_acc** - 测试准确率
- **dev_f1** - 验证集 F1 (观察过拟合)

#### CID 模块健康度
- **m_v_mean** - Vision mask 均值，应接近 0.4 (目标 rho)
- **m_t_mean** - Text mask 均值，应接近 0.6 (目标 rho_t)
- **tau** - 温度参数，应从 1.0 逐渐衰减到 0.4

#### 损失监控
- **train_loss** - 训练损失
- **test_loss** - 测试损失 (对比 train_loss 判断过拟合)

### 理想曲线特征

✅ **正常训练特征:**
- `macro_test_f1` 在 epoch 5-12 稳定增长
- `m_v_mean` 在 0.3-0.5 之间波动 (目标 0.4)
- `m_t_mean` 在 0.5-0.7 之间波动 (目标 0.6)
- `tau` 平滑衰减到 0.4

⚠️ **异常信号:**
- `train_loss` 持续下降但 `test_loss` 上升 → 过拟合
- `m_v_mean` 或 `m_t_mean` 接近 0 或 1 → CID 退化
- `macro_test_f1` 在后期不增长 → 可能欠拟合，尝试激进版

---

## 实验策略建议

### 阶段 1: 基线验证 (1-2 天)

```bash
# 运行推荐配置
bash train_single_stage.sh
```

**预期结果:** 相比原始 stage2 配置，macro F1 应有 2-5% 的提升

### 阶段 2: 根据结果调整

#### 如果 Macro F1 < 70%
→ 尝试**激进版** (更强监督)
```bash
bash train_exp_aggressive.sh
```

#### 如果 Macro F1 > 75% 但有过拟合迹象
→ 尝试**保守版** (降低过拟合风险)
```bash
bash train_exp_conservative.sh
```

#### 如果模型难以区分困难样本
→ 尝试**困难负样本版**
```bash
bash train_exp_hard_neg.sh
```

### 阶段 3: 精细调优 (可选)

基于最佳配置，手动微调：
- 调整 `dropout_rate` (0.1 → 0.15 减少过拟合)
- 调整 `weight_decay` (0.03 → 0.05 增强正则化)
- 尝试不同种子 (seed=42, 123, 456) 取平均

---

## 常见问题

### Q1: 训练到一半 OOM (显存不足) 怎么办?

**方案 1:** 减小 batch size
```bash
--train_batch_size 16  # 从 24 减小到 16
--dev_batch_size 16
```

**方案 2:** 启用梯度累积 (需修改 train.py，暂未实现)

### Q2: 如何更改 GPU 编号?

编辑训练脚本，修改第一行：
```bash
--device 1  # 改为你的 GPU 编号 (0, 1, 2, ...)
```

### Q3: 如何使用不同的随机种子?

编辑训练脚本，修改：
```bash
--seed 42  # 改为 123, 456, 789 等
```

建议运行 3 次不同种子取平均以降低随机性影响。

### Q4: 训练中途中断如何恢复?

当前脚本**不支持断点续训**。如需支持，需修改代码保存 optimizer 和 scheduler 状态。

---

## 文件清单

### 训练脚本
- ✅ `train_single_stage.sh` - 推荐基线配置
- ✅ `train_exp_aggressive.sh` - 激进版本
- ✅ `train_exp_conservative.sh` - 保守版本
- ✅ `train_exp_hard_neg.sh` - 困难负样本版本

### 修改的文件
- ✅ `model.py:633-634` - CID 参数 (rho=0.4, rho_t=0.6)

### 输出目录
- `../output_dir/single_stage/RCLMuFN/model.pt` - 基线模型
- `../output_dir/exp_aggressive/RCLMuFN/model.pt` - 激进版模型
- `../output_dir/exp_conservative/RCLMuFN/model.pt` - 保守版模型
- `../output_dir/exp_hard_neg/RCLMuFN/model.pt` - 困难负样本模型

---

## 优化原理总结

### 为什么增加 CID 损失权重?

原始权重 (`lambda_ratio=2e-3, lambda_itm=1.5e-3`) 可能过小，导致：
1. CID 模块未充分学习一致性/不一致性分解
2. mask 退化到全 0 或全 1 (失去分解能力)

增加到 `5e-3` 和 `3e-3` 后，CID 损失在总损失中占更大比重，强制模型学习有意义的分解。

### 为什么增加 CLIP 学习率?

CLIP 预训练在通用图文数据上，需要微调适配**仇恨言论检测**这个特定任务。
- 原始 `1e-6` 过小 → CLIP 几乎不更新
- 增加到 `5e-6` → 允许 CLIP 学习任务特定的图文表征

### 为什么增加 rho 和 rho_t?

MMSD 数据集中，**一致性特征**同样重要（例如明显的仇恨符号、攻击性文字）。
- 增加比例目标 → 鼓励模型提取更多一致性特征
- 平衡一致性和不一致性 → 更全面的特征表征

---

## 技术支持

如有问题，检查：
1. WandB 日志: `wandb/latest-run/`
2. 训练日志: 终端输出
3. 模型保存: `../output_dir/*/RCLMuFN/model.pt`

祝训练顺利! 🚀
