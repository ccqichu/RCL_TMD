# text_clean 数据集优化总结

## 数据集对比分析

### text_final vs text_clean

| 指标 | text_final | text_clean | 差异分析 |
|------|-----------|-----------|---------|
| **训练样本** | 19,816 | 19,557 | -259 (-1.3%) |
| **验证样本** | 2,410 | 2,387 | -23 (-1.0%) |
| **测试样本** | 2,409 | 2,373 | -36 (-1.5%) |
| **平均文本长度** | 64.9 字符 | **79.9 字符** | **+15 字符 (+23%)** ⚠️ |
| **训练集-仇恨比例** | 48.3% | **42.9%** | **-5.4% (更不平衡)** ⚠️ |
| **训练集-非仇恨** | 51.7% | 57.1% | +5.4% |

### 关键发现

1. **文本更长**: 平均长度从 64.9 增加到 79.9 (+23%)
   - 包含更多语义信息
   - 更容易过拟合
   - 需要更强的正则化

2. **类别更不平衡**: 仇恨样本从 48.3% 降到 42.9%
   - Macro F1 更关键（而非 Accuracy）
   - 少数类（仇恨言论）更难学习
   - 需要关注 recall@hate 指标

3. **样本略少**: 约减少 1-1.5%
   - 影响不大，但配合文本变长，训练难度增加

---

## 应用的优化策略

### 1. 模型参数调整 (model.py:634)

| 参数 | 原值 | text_clean 优化值 | 原因 |
|------|------|-----------------|------|
| `rho` (vision) | 0.3 | **0.4** | 保持（vision 不受文本长度影响） |
| `rho_t` (text) | 0.5 | **0.55** | **降低**，因为文本更长更复杂，允许更多不一致性特征 |

**原理**:
- text_clean 的文本包含更多细节，不一致性特征（如讽刺、隐喻）更分散
- 降低 `rho_t` 从 0.6 → 0.55，给予 CID 模块更多空间提取复杂的不一致性

---

### 2. 训练超参数调整

#### 针对 text_clean 的统一调整（所有脚本）

| 参数 | text_final 值 | text_clean 值 | 变化 | 原因 |
|------|--------------|--------------|------|------|
| **dropout_rate** | 0.1 | **0.15** | +50% | 防止长文本过拟合 |
| **weight_decay** | 0.03 | **0.05** | +67% | 增强 L2 正则化 |
| **warmup_proportion** | 0.2 | **0.15** | -25% | 更快进入主训练（长文本信息丰富） |
| **num_train_epochs** | 12 | **15** | +25% | 长文本需要更多训练轮次 |

---

### 3. 各实验配置对比

| 脚本 | Lambda Ratio | Lambda ITM | CLIP LR | Epochs | 适用场景 |
|------|-------------|-----------|---------|--------|---------|
| **train_single_stage.sh** (推荐) | 5e-3 | 3e-3 | 5e-6 | 15 | 平衡配置，首选 |
| **train_exp_aggressive.sh** | 1e-2 | 5e-3 | 1e-5 | 15 | 基线欠拟合时使用 |
| **train_exp_conservative.sh** | 3e-3 | 2e-3 | 3e-6 | 12 | 过拟合严重时使用 |
| **train_exp_hard_neg.sh** | 5e-3 | 3e-3 | 5e-6 | 15 | 困难样本效果差时使用 |

---

## 完整参数对比表

### train_single_stage.sh (推荐基线)

| 参数类别 | 参数 | text_final | text_clean | 变化 |
|---------|------|-----------|-----------|------|
| **数据** | text_name | text_final | **text_clean** | ✓ |
| | num_train_epochs | 12 | **15** | +25% |
| **优化器** | learning_rate | 3e-4 | 3e-4 | - |
| | clip_learning_rate | 5e-6 | 5e-6 | - |
| | weight_decay | 0.03 | **0.05** | +67% |
| | warmup_proportion | 0.2 | **0.15** | -25% |
| **正则化** | dropout_rate | 0.1 | **0.15** | +50% |
| | max_grad_norm | 3.0 | 3.0 | - |
| **CID损失** | lambda_ratio_end | 5e-3 | 5e-3 | - |
| | lambda_itm_end | 3e-3 | 3e-3 | - |
| | lambda_warmup_epochs | 3 | 3 | - |
| | lambda_ramp_epochs | 5 | 5 | - |
| **模型** | rho (vision) | 0.4 | 0.4 | - |
| | rho_t (text) | 0.6 | **0.55** | -8.3% |

---

## 预期效果与监控

### 预期性能提升

相比 text_final 的原始配置：
- **Macro F1**: 预期提升 **2-4%**
- **Test Accuracy**: 预期提升 **1-3%**
- **Hate Recall**: 由于类别不平衡，重点关注该指标

### WandB 监控重点

#### 关键指标 (优先级从高到低)

1. **macro_test_f1** - 主要优化目标 (类别不平衡)
2. **macro_test_recall** - 确保不漏检仇恨言论
3. **test_acc** - 整体准确率
4. **train_loss vs test_loss** - 过拟合监控

#### CID 健康度指标

| 指标 | 期望范围 | 说明 |
|------|---------|------|
| **m_v_mean** | 0.35-0.45 | Vision mask 均值，目标 0.4 |
| **m_t_mean** | 0.50-0.60 | Text mask 均值，目标 0.55 |
| **tau** | 1.0 → 0.4 | 温度参数，应平滑衰减 |

#### 异常信号

⚠️ **过拟合**:
- train_loss 持续下降，test_loss 上升或停滞
- 解决方案: 使用 conservative 脚本

⚠️ **欠拟合**:
- macro_test_f1 在 epoch 10 后仍低于 70%
- 解决方案: 使用 aggressive 脚本

⚠️ **CID 退化**:
- m_v_mean 或 m_t_mean 接近 0 或 1
- 解决方案: 增加 lambda 权重

---

## 使用建议

### 推荐工作流

#### 阶段 1: 基线验证 (1-2 天)

```bash
# 运行推荐配置
bash train_single_stage.sh
```

**监控要点**:
- Epoch 5-10: macro_test_f1 是否稳定增长
- Epoch 10-15: 是否出现过拟合 (test_loss 上升)

#### 阶段 2: 根据结果调整

| 情况 | 特征 | 下一步 |
|------|------|--------|
| **正常** | F1 > 72%, 无过拟合 | 完成！可尝试多种子平均 |
| **欠拟合** | F1 < 70%, loss 仍在下降 | `bash train_exp_aggressive.sh` |
| **过拟合** | train_loss << test_loss | `bash train_exp_conservative.sh` |
| **难样本差** | hate recall < 65% | `bash train_exp_hard_neg.sh` |

#### 阶段 3: 精细调优 (可选)

基于最佳配置的微调建议：

1. **多种子验证**:
   ```bash
   # 修改 --seed 参数运行 3 次
   --seed 42
   --seed 123
   --seed 456
   ```

2. **类别加权** (如果 hate recall 仍然很低):
   需要修改 train.py 添加 class_weight (暂未实现)

3. **学习率调优**:
   ```bash
   # 如果训练不稳定，尝试降低学习率
   --learning_rate 2e-4  # 从 3e-4 降低
   --clip_learning_rate 3e-6  # 从 5e-6 降低
   ```

---

## 关键改动总结

### ✅ 已完成的修改

1. **model.py:634** - `rho_t: 0.6 → 0.55`
2. **所有训练脚本**:
   - `text_name: text_final → text_clean`
   - `dropout_rate: 0.1 → 0.15`
   - `weight_decay: 0.03 → 0.05`
   - `warmup_proportion: 0.2 → 0.15`
   - `num_train_epochs: 12 → 15` (除 conservative 为 12)
   - `output_dir: 更新为 text_clean_* 目录`

### 📂 输出目录

| 脚本 | 输出目录 |
|------|---------|
| train_single_stage.sh | `../output_dir/text_clean_baseline/` |
| train_exp_aggressive.sh | `../output_dir/text_clean_aggressive/` |
| train_exp_conservative.sh | `../output_dir/text_clean_conservative/` |
| train_exp_hard_neg.sh | `../output_dir/text_clean_hard_neg/` |

---

## 常见问题

### Q1: 为什么 rho_t 降低而不是增加？

**A**: text_clean 的文本更长更复杂，不一致性特征（如讽刺、隐喻）更分散在长文本中。降低 `rho_t` 允许 CID 模块标记更多 token 为不一致性，从而捕获这些分散的复杂模式。

### Q2: 为什么增加 dropout 而不是减少？

**A**: 长文本包含更多信息，模型更容易记忆训练样本的细节导致过拟合。增加 dropout 强制模型学习更鲁棒的特征，而不是依赖特定的 token 组合。

### Q3: 15 epochs 会不会太长？

**A**: 对于 text_clean：
- 文本长度 +23%，语义复杂度更高
- 类别不平衡更严重，少数类需要更多训练
- 实际上 15 epochs 对应约 **1,224 steps/epoch × 15 = 18,360 steps**，仍然适中

如果出现过拟合，可以提前停止或使用 conservative 脚本（12 epochs）。

### Q4: 如何选择 GPU?

所有脚本默认使用 `--device 1`，如需更改：
```bash
# 编辑脚本，修改第 22 行（或 15 行）
--device 3  # 改为你的 GPU 编号
```

---

## 文件清单

### 修改的文件
- ✅ `model.py:634` - CID 参数 (rho_t=0.55)

### 更新的训练脚本
- ✅ `train_single_stage.sh` - 推荐基线（text_clean 优化）
- ✅ `train_exp_aggressive.sh` - 激进版（text_clean 优化）
- ✅ `train_exp_conservative.sh` - 保守版（text_clean 优化）
- ✅ `train_exp_hard_neg.sh` - 困难负样本（text_clean 优化）

### 新增文档
- ✅ `TRAINING_GUIDE.md` - 通用训练指南
- ✅ `TEXT_CLEAN_OPTIMIZATION.md` - 本文档

---

## 立即开始训练

```bash
# 1. 确认所有脚本已更新
ls -lh train_*.sh

# 2. 运行推荐配置（首选）
bash train_single_stage.sh

# 预计时间: 20-25 小时
# 输出: ../output_dir/text_clean_baseline/RCLMuFN/model.pt
```

祝训练顺利！ 🚀
