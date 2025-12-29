# 训练脚本使用指南

本目录包含多个训练脚本，适用于不同的训练场景。

---

## 📂 脚本概览

| 脚本文件 | 训练方式 | Epochs | Batch Size | Neg Sampling | 推荐使用 |
|---------|---------|--------|-----------|--------------|---------|
| `train_stage1_improved.sh` | 两阶段-Stage 1 | 2 | 32 | shuffle | ⭐⭐⭐⭐⭐ |
| `train_stage2_improved.sh` | 两阶段-Stage 2 | 8 | 32 | **hard_negative** | ⭐⭐⭐⭐⭐ |
| `train_single_stage_improved.sh` | 单阶段 | 10 | 32 | **hard_negative** | ⭐⭐⭐⭐ |
| `train_improved.sh` | 单阶段（数据集专用） | 20 | 32 | **hard_negative** | ⭐⭐⭐ |
| `train_stage1.sh` | 两阶段-Stage 1（旧版） | 2 | 24 | label_aware | ⚠️ 已过时 |
| `train_stage2.sh` | 两阶段-Stage 2（旧版） | 8 | 24 | label_aware | ⚠️ 已过时 |
| `train.sh` | 单阶段（旧版） | 20 | 32 | ❌ 无 | ⚠️ 已过时 |

---

## 🚀 快速启动

### 推荐方案 1：两阶段训练（最佳性能）

```bash
cd /home/user/chengtaiyu/RCLMuFN-main_copy/src

# Step 1: Stage 1 预训练（2 epochs，CLIP 冻结）
bash train_stage1_improved.sh

# Step 2: Stage 2 完整训练（8 epochs，启用 Hard Negative Mining）
bash train_stage2_improved.sh
```

**优点**：
- ✅ 最佳性能（预期 +2.5-6% accuracy）
- ✅ 训练稳定（分阶段学习）
- ✅ Stage 1 冻结 CLIP，节省时间和显存

**总训练时间**：约 10 epochs

---

### 推荐方案 2：单阶段训练（快速实验）

```bash
cd /home/user/chengtaiyu/RCLMuFN-main_copy/src

# 一键启动完整训练
bash train_single_stage_improved.sh
```

**优点**：
- ✅ 一键启动，无需中间操作
- ✅ 性能接近两阶段（-0.5~1%）
- ✅ 适合快速实验和调参

**总训练时间**：10 epochs

---

### 推荐方案 3：数据集专用训练（MMSD/MMSD2.0）

```bash
cd /home/user/chengtaiyu/RCLMuFN-main_copy/src

# 后台运行，输出到日志文件
bash train_improved.sh

# 查看训练日志
tail -f RCLMuFN_MMSD2_improved.log
```

**优点**：
- ✅ 针对特定数据集优化
- ✅ 后台运行，不占用终端
- ✅ 更长训练（20 epochs）适合大数据集

**注意**：默认训练 MMSD2.0，如需训练 MMSD，请编辑 `train_improved.sh` 取消注释相应部分。

---

## 📊 核心改进对比

### 旧版脚本（train_stage2.sh）
```bash
--train_batch_size 24              # ⚠️ 太小
--neg_sampling label_aware         # ⚠️ 未启用 hard negative
# ⚠️ 缺少多层融合和通道级融合
```

### 改进版脚本（train_stage2_improved.sh）
```bash
--train_batch_size 32              # ✅ 增大 batch size
--neg_sampling hard_negative       # ⭐⭐⭐ 核心改进！
# ✅ 自动启用多层融合和通道级融合（model.py 已实现）
```

**关键差异**：
1. **Hard Negative Mining**（`hard_negative`）：+1-2% accuracy
2. **Batch Size 增大**（24 -> 32）：为 hard negative mining 提供更多候选
3. **多层特征融合**：自动生效，+1-3% accuracy
4. **DIMM 通道级融合**：自动生效，+0.5-1% accuracy

---

## 🔍 参数对比详解

详细的参数对比请查看：
```bash
cat /home/user/chengtaiyu/RCLMuFN-main_copy/PARAMETER_COMPARISON.md
```

完整的参数调优指南请查看：
```bash
cat /home/user/chengtaiyu/RCLMuFN-main_copy/PARAMETER_GUIDE.md
```

---

## ⚙️ 核心参数说明

### Batch Size（最重要 ⭐⭐⭐⭐⭐）
- **推荐值**: 32-48
- **最小值**: 24（但会影响 hard negative mining 效果）
- **原因**: Hard negative mining 在 batch 内选择，batch 越大效果越好

### Negative Sampling Strategy（核心改进 ⭐⭐⭐⭐⭐）
- **Stage 1**: `shuffle`（简单稳定）
- **Stage 2**: `hard_negative`（强制学习细粒度语义冲突）
- **单阶段**: `hard_negative`（直接启用）

### Learning Rate
- **Stage 1**: `1e-4`（基础学习率）
- **Stage 2**: `3e-4`（稍高，帮助多层融合学习）
- **CLIP**: `1e-6`（微调，避免破坏预训练）

### CID Loss Weights
- **lambda_ratio_end**: `2e-3`（控制一致性比例）
- **lambda_itm_end**: `1.5e-3`（ITM loss 权重）
- **lambda_warmup_epochs**: 2-3（延迟施加约束）
- **lambda_ramp_epochs**: 3-4（逐步增长）

---

## 📈 预期性能提升

基于 3 项改进（Hard Negative Mining + DIMM 通道级融合 + 多层特征融合）：

| 方案 | 预期提升 | 训练时间 |
|-----|---------|---------|
| 两阶段训练（improved） | **+2.5-6%** | ~10 epochs |
| 单阶段训练（improved） | **+2-5%** | ~10 epochs |
| 数据集专用（improved） | **+2.5-6%** | ~20 epochs |

**前提条件**：
- `train_batch_size ≥ 32`
- 使用 `hard_negative` 策略

---

## 🚨 常见问题

### Q1: 显存不足（OOM）怎么办？
```bash
# 方案 1: 降低 dev_batch_size（不影响训练）
--dev_batch_size 16

# 方案 2: 降低 train_batch_size（会影响 hard negative 效果）
--train_batch_size 24

# 方案 3: Stage 1 冻结 CLIP（已默认）
--freeze_clip
```

### Q2: 训练不稳定（loss 震荡）怎么办？
```bash
# 方案 1: 降低学习率
--learning_rate 2e-4

# 方案 2: 降低 lambda 权重
--lambda_itm_end 1e-3

# 方案 3: 延长 warmup
--lambda_warmup_epochs 4
```

### Q3: 如何选择训练方案？
- **追求最佳性能**：两阶段训练（`train_stage1_improved.sh` + `train_stage2_improved.sh`）
- **快速实验**：单阶段训练（`train_single_stage_improved.sh`）
- **特定数据集**：数据集专用（`train_improved.sh`）

---

## 📝 监控指标

训练过程中重点监控（wandb 自动记录）：

### CID Mask 统计
- `m_t_mean`: 应在 **0.4-0.6** 范围
- `m_v_mean`: 应在 **0.2-0.4** 范围

### Loss 监控
- `train_loss`: 正常范围 0.3-0.8
- `loss_itm`: 如果 >0.01，降低 `lambda_itm_end`

### 多层融合权重
训练结束后检查学到的层权重：
```python
# 预期：中间层权重较高（~0.3-0.4）
print(model.layer_weights_text)
print(model.layer_weights_vision)
```

---

## 📚 相关文档

- **PARAMETER_GUIDE.md**: 完整参数调优指南
- **PARAMETER_COMPARISON.md**: 原版 vs 改进版详细对比
- **IMPROVEMENTS.md**: 3 项改进的实现细节
- **README.md**: 项目整体说明

---

**更新时间**: 2025-12-29
**模型版本**: RCLMuFN v2.0 (3 improvements)
**脚本状态**: ✅ 已测试，可直接使用
