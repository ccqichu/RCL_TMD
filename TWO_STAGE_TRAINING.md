# 双阶段训练方案使用指南

## 概述

双阶段训练方案将模型训练分为两个阶段：
1. **Stage 1（预热阶段）**：冻结 CLIP，学习基础融合和分类
2. **Stage 2（完整训练）**：解冻 CLIP，启用 CID 损失，完整优化

## 为什么使用双阶段训练？

### 架构优势
- **可退回设计（alpha=0）**：CID-DIMM 通过 `alpha` 参数可以渐进式引入
- **多损失复杂性**：L_fuse + L_ratio + L_itm 同时优化可能相互干扰
- **温度退火需求**：CID 的温度退火需要稳定的初始特征表示
- **CLIP 冻结收益**：Stage 1 冻结 CLIP 节省 82.5% 参数优化成本

### 预期效果
- **更稳定的训练**：分阶段引入复杂度，避免早期震荡
- **更好的性能**：预期相比单阶段训练提升 **2-5% Acc**
- **节省资源**：Stage 1 显存占用更低，训练速度更快

---

## Stage 1：预热阶段（2 epochs）

### 目标
- 让模型学会基础的多模态融合和分类
- 为 CID-DIMM 提供稳定的初始化

### 关键配置
```bash
--freeze_clip                    # 冻结 CLIP（151M 参数）
--learning_rate 1e-4             # 较小学习率
--clip_learning_rate 0           # CLIP 不更新
--lambda_ratio_end 0.0           # 不启用 L_ratio
--lambda_itm_end 0.0             # 不启用 L_itm
--num_train_epochs 2             # 仅训练 2 epochs
```

### 参数统计
- **Total parameters**: 183,385,864
- **Trainable parameters**: 32,108,551 (17.5%)
- **Frozen parameters**: 151,277,313 (82.5%)

### 运行方式
```bash
cd /home/user/chengtaiyu/RCLMuFN-main_copy/src
bash train_stage1.sh
```

### Checkpoint 保存位置
```
../output_dir/stage1/RCLMuFN/model.pt
```

---

## Stage 2：完整训练（8 epochs）

### 目标
- 启动 CID 监督（L_ratio + L_itm）
- CLIP 微调以配合 CID-DIMM
- 温度退火生效

### 关键配置
```bash
--resume_from ../output_dir/stage1/RCLMuFN/model.pt  # 加载 Stage 1 checkpoint
# 不使用 --freeze_clip（解冻 CLIP）
--learning_rate 3e-4             # 正常学习率
--clip_learning_rate 1e-6        # CLIP 微调学习率
--lambda_ratio_end 2e-3          # 启用 L_ratio
--lambda_itm_end 1.5e-3          # 启用 L_itm
--lambda_warmup_epochs 2         # 前 2 epochs 保持 lambda=0
--lambda_ramp_epochs 3           # 接下来 3 epochs 线性增长
--num_train_epochs 8             # 训练 8 epochs
```

### Lambda 权重调度表

| Epoch | Lambda_ratio | Lambda_ITM | Phase      |
|-------|--------------|------------|------------|
| 0     | 0.000000     | 0.000000   | Warmup     |
| 1     | 0.000000     | 0.000000   | Warmup     |
| 2     | 0.000667     | 0.000500   | Ramp       |
| 3     | 0.001333     | 0.001000   | Ramp       |
| 4     | 0.002000     | 0.001500   | Ramp       |
| 5     | 0.002000     | 0.001500   | Full       |
| 6     | 0.002000     | 0.001500   | Full       |
| 7     | 0.002000     | 0.001500   | Full       |

### 参数统计
- **Total parameters**: 183,385,864
- **Trainable parameters**: 183,385,864 (100.0%)

### 运行方式
```bash
cd /home/user/chengtaiyu/RCLMuFN-main_copy/src
bash train_stage2.sh
```

### Checkpoint 保存位置
```
../output_dir/stage2/RCLMuFN/model.pt
```

---

## 完整训练流程

### 1. 测试配置（可选）
```bash
python test_two_stage.py
```
这会验证：
- Stage 1/2 参数配置
- CLIP 冻结/解冻逻辑
- Lambda 调度正确性
- Checkpoint 加载机制

### 2. 运行 Stage 1
```bash
bash train_stage1.sh
```
预期耗时：~2-3 小时（取决于硬件）

### 3. 检查 Stage 1 结果
```bash
# 查看 wandb 日志
# 检查 dev_acc 是否合理（应该 > 60%）
```

### 4. 运行 Stage 2
```bash
bash train_stage2.sh
```
预期耗时：~12-16 小时（取决于硬件）

### 5. 最终评估
最佳模型保存在：
```
../output_dir/stage2/RCLMuFN/model.pt
```

---

## 训练监控要点

### Stage 1 监控
- ✅ **loss_fuse** 下降平稳
- ✅ **dev_acc** 稳定增长
- ✅ **alpha** 自然增长（有梯度但无监督）
- ⚠️ **避免过拟合**（dev_acc 应该 > 60%）

### Stage 2 监控
- ✅ **lambda_ratio/lambda_itm** 按调度增长
- ✅ **tau** 逐 epoch 衰减（1.0 → 0.4）
- ✅ **CID stats**：
  - m_t_mean: 0.3-0.5
  - m_v_mean: 0.2-0.4
- ✅ **loss 分解**：
  - loss_fuse: 主损失
  - loss_ratio: < 0.1
  - loss_itm: < 0.2

---

## 常见问题

### Q1: 如果 Stage 1 训练效果不好怎么办？
可能原因：
- Learning rate 过大/过小 → 调整 `--learning_rate`
- Batch size 不合适 → 调整 `--train_batch_size`
- Warmup 不足 → 增加 `--warmup_proportion`

### Q2: 如果显存不足怎么办？
解决方案：
- 减小 batch_size（24 → 16 或 12）
- 使用梯度累积（需要修改 train.py）
- 减少 num_workers（4 → 2）

### Q3: 可以跳过 Stage 1 直接训练 Stage 2 吗？
不建议。Stage 1 的预热对 CID-DIMM 稳定性至关重要。但如果坚持，可以：
```bash
python main.py \
  --num_train_epochs 10 \
  --train_batch_size 24 \
  --learning_rate 3e-4 \
  --lambda_ratio_end 2e-3 \
  --lambda_itm_end 1.5e-3 \
  --lambda_warmup_epochs 3 \
  --lambda_ramp_epochs 4 \
  # 不使用 --freeze_clip 和 --resume_from
```

### Q4: 如何从 Stage 2 中断的训练恢复？
修改 `train_stage2.sh` 中的 checkpoint 路径：
```bash
--resume_from ../output_dir/stage2/RCLMuFN/model.pt
```

---

## 代码修改总结

### 修改文件
1. **main.py** (lines 58-62, 121-129)
   - 添加 `--resume_from` 和 `--freeze_clip` 参数
   - 添加 checkpoint 加载逻辑

2. **train.py** (lines 43-54, 86-95)
   - 添加 CLIP 冻结逻辑
   - 修改 optimizer 创建逻辑（冻结时不优化 CLIP）

### 新增文件
1. **train_stage1.sh**: Stage 1 训练脚本
2. **train_stage2.sh**: Stage 2 训练脚本
3. **test_two_stage.py**: 配置测试脚本
4. **TWO_STAGE_TRAINING.md**: 使用文档（本文件）

---

## 预期性能提升

| 训练方式          | 预期 Acc | 训练稳定性 | 显存占用 | 训练时间 |
|-------------------|---------|-----------|---------|---------|
| 单阶段端到端       | 基线     | ⚠️ 中等    | 高       | 短       |
| **双阶段（推荐）** | **+2-5%** | ✅ 稳定    | Stage1低 | 中等     |

---

## 下一步优化方向

如果双阶段训练效果仍不理想，可以考虑：

1. **三阶段训练**：
   - Stage 1: 冻结 CLIP，无 CID 损失
   - Stage 2: 冻结 CLIP，启用 CID 损失
   - Stage 3: 解冻 CLIP，完整训练

2. **更精细的 Lambda 调度**：
   - 使用 cosine 调度替代 linear
   - 延长 warmup epochs

3. **数据增强**：
   - 文本：回译、同义词替换
   - 图像：随机裁剪、颜色抖动

4. **Ensemble**：
   - 训练多个不同 seed 的模型
   - Soft voting 融合预测结果

---

**祝训练顺利！🚀**
