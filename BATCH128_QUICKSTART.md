# Batch Size 128 实验快速启动指南

## 📋 概述

这套脚本将自动运行 **3种优化方法 × 3个随机种子 = 9次完整训练**，帮你找到指标最高的配置。

### 三种方法
1. **Conservative（保守）**: 稳妥的学习率和参数，成功率最高
2. **Aggressive（激进）**: 更激进的学习率，预期性能最优
3. **Extreme（极致）**: 最激进配置，冲击最高指标

### 种子选择
- **42**: 历史最佳F1 (0.8485)
- **96**: 历史最佳Acc (0.8506)
- **100**: 第三名 (0.8470)

---

## 🚀 快速开始（3步）

### Step 1: 启动训练

```bash
cd /home/user/chengtaiyu/RCLMuFN-main_copy/src

# 后台运行，即使关闭终端也会继续训练
nohup bash train_batch128_all_methods.sh > train_batch128.log 2>&1 &

# 记住进程ID
echo $! > train_batch128.pid
```

**预计耗时**: 30-50小时（取决于GPU）
- Conservative: ~3-4小时/次 × 3 = 9-12小时
- Aggressive: ~4-5小时/次 × 3 = 12-15小时
- Extreme: ~5-7小时/次 × 3 = 15-21小时

### Step 2: 监控训练

```bash
# 实时监控GPU和训练进度
watch -n 10 bash monitor_training.sh

# 或者查看日志
tail -f train_batch128.log

# 或者查看主日志
tail -f ../output_dir/batch128_experiments/master_training_log_*.log
```

### Step 3: 分析结果

训练完成后：

```bash
# 自动分析所有结果
python analyze_batch128_results.py

# 查看生成的摘要
cat ../output_dir/batch128_experiments/summary_*.txt

# 查看CSV格式结果
cat ../output_dir/seed_runs/batch128_comparison.csv
```

---

## 📂 输出文件结构

```
output_dir/
├── batch128_experiments/
│   ├── conservative_seed42/
│   │   ├── RCLMuFN/
│   │   │   └── model.pt          # 训练的模型
│   │   └── training.log           # 训练日志
│   ├── conservative_seed96/
│   ├── conservative_seed100/
│   ├── aggressive_seed42/
│   ├── aggressive_seed96/
│   ├── aggressive_seed100/
│   ├── extreme_seed42/
│   ├── extreme_seed96/
│   ├── extreme_seed100/
│   ├── master_training_log_*.log  # 主日志（包含所有运行）
│   └── summary_*.txt              # 实验总结
│
└── seed_runs/
    ├── conservative_seed42.json   # 详细指标
    ├── conservative_seed96.json
    ├── conservative_seed100.json
    ├── aggressive_seed42.json
    ├── aggressive_seed96.json
    ├── aggressive_seed100.json
    ├── extreme_seed42.json
    ├── extreme_seed96.json
    ├── extreme_seed100.json
    └── batch128_comparison.csv    # CSV对比表
```

---

## 📊 结果示例

运行 `python analyze_batch128_results.py` 后会显示：

### 1. 详细对比表
```
Method          Seed   Acc     Macro F1  Macro P   Macro R   Epoch
conservative    42     0.8530  0.8510    0.8505    0.8515    4
conservative    96     0.8525  0.8505    0.8500    0.8510    3
conservative    100    0.8520  0.8500    0.8495    0.8505    5
aggressive      42     0.8565  0.8545    0.8540    0.8550    5
aggressive      96     0.8560  0.8540    0.8535    0.8545    4
...
```

### 2. 方法统计
```
Method          Avg Acc    Std Acc    Avg F1     Std F1     Avg Epoch
conservative    0.8525     0.0005     0.8505     0.0005     4.0
aggressive      0.8563     0.0003     0.8543     0.0003     4.7
extreme         0.8580     0.0010     0.8560     0.0010     6.3
```

### 3. 最佳配置
```
🏆 最佳单次配置:
   Method: EXTREME
   Seed: 42
   Test Accuracy: 0.8595
   Macro F1: 0.8575
   Best Epoch: 6
   Model Path: ../output_dir/batch128_experiments/extreme_seed42/RCLMuFN/model.pt
```

---

## 🛠️ 高级用法

### 只运行特定方法

修改 `train_batch128_all_methods.sh` 中的循环：

```bash
# 只运行 aggressive 方法
for method in aggressive; do
    ...
done
```

### 只运行特定种子

```bash
# 只使用 seed=42
SEEDS=(42)
```

### 修改训练参数

在 `train_batch128_all_methods.sh` 中找到对应的方法函数（如 `run_aggressive`），修改参数：

```bash
run_aggressive() {
    local seed=$1

    train_model \
        "aggressive" \
        $seed \
        1.3e-3 \        # ← 修改学习率
        1e-6 \
        0.25 \
        12 \
        2.5e-3 \
        2e-3 \
        ...
}
```

### 添加新方法

```bash
# 在脚本中添加新函数
run_custom() {
    local seed=$1
    train_model \
        "custom" \
        $seed \
        1.1e-3 \  # 自定义学习率
        ...
}

# 在主循环中添加
for method in conservative aggressive extreme custom; do
    ...
done
```

---

## 🔧 故障排除

### 问题1: 显存不足

**症状**: `CUDA out of memory`

**解决方案**:
1. 降低batch size（需要修改脚本中的 `--train_batch_size 128` 为 `96` 或 `64`）
2. 减少 `--layers` 从 4 降到 3
3. 使用梯度累积（需要修改train.py）

### 问题2: 训练中断

**恢复方法**:
```bash
# 查看已完成的实验
ls ../output_dir/seed_runs/

# 手动删除未完成的实验目录
rm -rf ../output_dir/batch128_experiments/aggressive_seed96

# 修改脚本跳过已完成的实验
# 在对应的 for 循环中添加检查：
if [ -f "${RESULTS_DIR}/aggressive_seed96.json" ]; then
    log_info "Skipping aggressive_seed96 (already completed)"
    continue
fi
```

### 问题3: 某个seed失败

**不影响其他实验**：脚本会继续运行剩余配置，最后统计时会标记失败项。

### 问题4: 进程被kill

**重新启动**:
```bash
# 查看进程是否还在
ps aux | grep train_batch128

# 如果没有，重新启动
nohup bash train_batch128_all_methods.sh > train_batch128.log 2>&1 &
```

---

## 📈 预期性能提升

相比 batch_size=96 的最佳结果 (Acc=0.8506, F1=0.8485)：

| 方法 | 预期Acc | 提升 | 预期F1 | 提升 |
|------|---------|------|--------|------|
| Conservative | 85.3-85.8% | +0.2-0.7% | 84.8-85.2% | +0.0-0.4% |
| Aggressive | 85.5-86.0% | +0.4-0.9% | 85.0-85.5% | +0.2-0.7% |
| Extreme | 85.8-86.5% | +0.7-1.4% | 85.5-86.0% | +0.7-1.2% |

**最乐观预期**: Extreme方法可能达到 **86.5% Acc / 86.0% F1**

---

## 🎯 下一步优化

如果三种方法都完成后，可以进一步优化：

### 1. Ensemble（推荐）
```bash
# 融合top-3配置的预测
# 预期提升：+0.3-0.5%
python ensemble_models.py \
    --models extreme_seed42 aggressive_seed42 extreme_seed96 \
    --weights 0.4 0.3 0.3
```

### 2. 更大Batch Size
```bash
# 如果显存充足，尝试 batch=160 或 192
# 修改脚本中的 --train_batch_size
```

### 3. 学习率网格搜索
```bash
# 在最佳方法基础上，尝试 lr ± 20%
# 例如 aggressive 的 1.2e-3 → [1.0e-3, 1.1e-3, 1.2e-3, 1.3e-3, 1.4e-3]
```

### 4. 双阶段训练
```bash
# 结合两阶段训练策略
bash train_stage1.sh  # 2 epochs, frozen CLIP
bash train_stage2.sh  # 8 epochs, 使用aggressive参数
```

---

## 📞 快速命令速查

```bash
# 启动训练
nohup bash train_batch128_all_methods.sh > train.log 2>&1 &

# 监控进度
watch -n 10 bash monitor_training.sh

# 查看日志
tail -f train.log

# 停止训练
kill $(cat train_batch128.pid)

# 分析结果
python analyze_batch128_results.py

# 查看GPU
watch -n 1 nvidia-smi

# 查看完成数量
ls ../output_dir/seed_runs/*_seed*.json | wc -l

# 检查失败的运行
grep -i "error\|failed" ../output_dir/batch128_experiments/master_training_log_*.log
```

---

## ✅ 检查清单

在启动训练前确认：

- [ ] GPU可用且显存充足（至少16GB）
- [ ] 磁盘空间充足（至少50GB）
- [ ] Python环境正确（transformers, torch, sklearn等）
- [ ] CLIP模型已下载到 `/home/user/chengtaiyu/models/clip-vit-base-patch32`
- [ ] 数据集路径正确
- [ ] 训练脚本有执行权限 (`chmod +x`)
- [ ] 使用 `nohup` 以防止终端关闭中断训练

---

## 🎉 预祝实验成功！

记得定期查看训练进度，大约30-50小时后你将获得**9个完整训练的模型**和详细的性能对比报告。

如有问题，检查：
1. `train_batch128.log` - 主输出日志
2. `master_training_log_*.log` - 详细训练日志
3. 各个方法的 `training.log` - 单次训练日志

**Good luck! 🚀**
