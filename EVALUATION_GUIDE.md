# 模型评估与对比分析指南

本指南介绍如何使用评估脚本测试训练好的模型权重，并生成详细的性能对比报告。

---

## 📁 文件说明

| 文件 | 功能 | 类型 |
|------|------|------|
| `evaluate.py` | 单个模型评估脚本 | Python |
| `batch_evaluate.sh` | 批量模型评估脚本 | Bash |
| `compare_results.py` | 结果对比分析脚本 | Python |

---

## 🚀 快速开始

### 方式 1: 批量评估所有模型（推荐）

```bash
# 自动评估所有训练好的模型
bash batch_evaluate.sh
```

**此命令会：**
- ✓ 自动搜索 `../output_dir/` 下的所有模型
- ✓ 在测试集和验证集上评估每个模型
- ✓ 生成详细的指标报告和混淆矩阵
- ✓ 将结果保存到 `./eval_results/` 目录

**预计时间：** 每个模型约 5-10 分钟

---

### 方式 2: 评估单个模型

```bash
python evaluate.py \
    --checkpoint ../output_dir/text_clean_baseline/RCLMuFN/model.pt \
    --text_name text_clean \
    --device 0 \
    --splits test,valid \
    --batch_size 32 \
    --output_dir ./eval_results \
    --save_predictions
```

**参数说明：**
- `--checkpoint`: 模型权重文件路径 (必需)
- `--text_name`: 数据集名称 (text_clean 或 text_final)
- `--device`: GPU 编号
- `--splits`: 评估的数据集划分 (test, valid, train)
- `--save_predictions`: 保存预测结果

---

### 方式 3: 对比分析所有结果

```bash
# 先运行批量评估
bash batch_evaluate.sh

# 然后生成对比报告
python compare_results.py --eval_dir ./eval_results
```

**生成的对比文件：**
- `comparison_report_*.txt` - 文本格式对比报告
- `comparison_test_*.csv` - 测试集结果 CSV
- `comparison_valid_*.csv` - 验证集结果 CSV
- `comparison_test.png` - 测试集可视化对比图
- `comparison_valid.png` - 验证集可视化对比图

---

## 📊 评估输出详解

### 1. 单个模型评估输出

每个模型会生成 3 个文件：

#### (1) JSON 结果文件 (`*_results.json`)

```json
{
  "checkpoint": "../output_dir/text_clean_baseline/RCLMuFN/model.pt",
  "dataset": "text_clean",
  "timestamp": "20260114_220000",
  "splits": {
    "test": {
      "num_samples": 2373,
      "accuracy": 0.7823,
      "macro": {
        "precision": 0.7801,
        "recall": 0.7789,
        "f1": 0.7795
      },
      "micro": {
        "precision": 0.7823,
        "recall": 0.7823,
        "f1": 0.7823
      },
      "per_class": {
        "Non-Hate": {
          "precision": 0.8234,
          "recall": 0.8512,
          "f1": 0.8371,
          "support": 1448
        },
        "Hate": {
          "precision": 0.7368,
          "recall": 0.7065,
          "f1": 0.7213,
          "support": 925
        }
      },
      "confusion_matrix": [[1232, 216], [272, 653]]
    }
  }
}
```

#### (2) 文本报告 (`*_report.txt`)

```
======================================================================
  RCL_TMD MODEL EVALUATION REPORT
======================================================================

Checkpoint: ../output_dir/text_clean_baseline/RCLMuFN/model.pt
Dataset: text_clean
Evaluation Time: 20260114_220000
Device: cuda:0

======================================================================
  TEST SET (2373 samples)
======================================================================

Overall Accuracy: 0.7823

Micro Average:
  Precision: 0.7823
  Recall:    0.7823
  F1-Score:  0.7823

Macro Average:
  Precision: 0.7801
  Recall:    0.7789
  F1-Score:  0.7795

Per-Class Metrics:
----------------------------------------------------------------------
Class           Precision       Recall    F1-Score      Support
----------------------------------------------------------------------
Non-Hate          0.8234       0.8512       0.8371         1448
Hate              0.7368       0.7065       0.7213          925

Confusion Matrix:
                Predicted
              Non-Hate  Hate
Actual Non-Hate   1232   216
       Hate        272   653
```

#### (3) 混淆矩阵图 (`*_confusion_matrix.png`)

可视化的混淆矩阵热力图，显示预测和真实标签的分布。

---

### 2. 对比分析输出

#### 对比报告示例 (`comparison_report_*.txt`)

```
====================================================================================================
  COMPREHENSIVE MODEL COMPARISON REPORT
====================================================================================================

Generated: 2026-01-14 22:00:00
Total Experiments: 4

====================================================================================================
  TEST SET COMPARISON
====================================================================================================

OVERALL METRICS:
----------------------------------------------------------------------------------------------------
Experiment                    Accuracy    Macro F1    Macro Precision    Macro Recall
----------------------------------------------------------------------------------------------------
text_clean_baseline             0.7823      0.7795           0.7801          0.7789
text_clean_aggressive           0.7891      0.7867           0.7875          0.7859
text_clean_conservative         0.7756      0.7728           0.7735          0.7721
text_clean_hard_neg             0.7845      0.7819           0.7825          0.7813

PER-CLASS METRICS:
----------------------------------------------------------------------------------------------------
Experiment                    Hate F1    Hate Precision    Hate Recall    Non-Hate F1
----------------------------------------------------------------------------------------------------
text_clean_baseline            0.7213        0.7368           0.7065         0.8371
text_clean_aggressive          0.7289        0.7445           0.7138         0.8445
text_clean_conservative        0.7145        0.7292           0.7002         0.8311
text_clean_hard_neg            0.7237        0.7390           0.7089         0.8401

BEST MODELS:
----------------------------------------------------------------------------------------------------
  Best Macro F1:       text_clean_aggressive (0.7867)
  Best Accuracy:       text_clean_aggressive (0.7891)
  Best Hate Recall:    text_clean_aggressive (0.7138)
  Best Hate F1:        text_clean_aggressive (0.7289)
----------------------------------------------------------------------------------------------------
```

#### 可视化对比图 (`comparison_test.png`)

包含 4 个子图：
1. **Macro Average Metrics** - 宏平均指标对比
2. **Overall Performance** - 整体性能 (Accuracy, Micro F1)
3. **Per-Class F1-Score** - 各类别 F1 分数
4. **Hate Class Metrics** - 仇恨类指标详情

---

## 📈 关键指标解读

### 优先级排序

| 指标 | 重要性 | 说明 |
|------|--------|------|
| **Macro F1** | ⭐⭐⭐ | **最重要** - 考虑类别不平衡，适合 MMSD 数据集 |
| **Hate Recall** | ⭐⭐⭐ | **最重要** - 确保不漏检仇恨言论 |
| **Hate F1** | ⭐⭐ | 仇恨类的综合性能 |
| **Macro Recall** | ⭐⭐ | 各类别召回率的平均 |
| **Accuracy** | ⭐ | 整体准确率（受类别不平衡影响） |

### 指标含义

#### Macro vs Micro 平均

- **Macro Average**: 先计算每个类别的指标，再取平均
  - 优点：对少数类（Hate）更公平
  - 适用：类别不平衡的数据集（如 text_clean）

- **Micro Average**: 将所有样本混合计算
  - 等同于 Accuracy
  - 受多数类（Non-Hate）主导

#### Per-Class 指标

- **Hate Recall**: 真实仇恨言论中被正确识别的比例
  - 高 recall → 不漏检
  - 低 recall → 存在漏报风险

- **Hate Precision**: 预测为仇恨的样本中真正是仇恨的比例
  - 高 precision → 不误报
  - 低 precision → 存在误报风险

---

## 🔍 结果分析示例

### 场景 1: 选择最佳模型

**目标**: 找到综合性能最好的模型

**步骤**:
1. 运行批量评估: `bash batch_evaluate.sh`
2. 生成对比报告: `python compare_results.py`
3. 查看 `comparison_report_*.txt`
4. 优先看 **Test Set** 的 **Macro F1** 和 **Hate Recall**

**决策规则**:
- 如果 Macro F1 最高 → 选该模型
- 如果 Hate Recall 显著更高（+3%以上）→ 考虑牺牲一点 F1 换取更好的召回

---

### 场景 2: 诊断模型问题

#### 问题 1: Hate Recall 很低（< 65%）

**可能原因**:
- 类别不平衡导致模型偏向多数类
- 训练时 lambda 权重过小

**解决方案**:
- 使用 `train_exp_aggressive.sh` (更强的 CID 监督)
- 增加 `lambda_itm_end` 权重

#### 问题 2: Train vs Test 性能差距大

**可能原因**:
- 过拟合

**解决方案**:
- 使用 `train_exp_conservative.sh`
- 检查 WandB 日志中的 `train_loss` vs `test_loss`

#### 问题 3: Non-Hate F1 很高但 Hate F1 很低

**可能原因**:
- 模型学习到了"偷懒"策略，倾向预测多数类

**解决方案**:
- 检查混淆矩阵，看 Hate 样本是否大量被误分为 Non-Hate
- 增加 Hate 类的损失权重（需修改代码）

---

## 🛠️ 高级用法

### 1. 只评估测试集

```bash
python evaluate.py \
    --checkpoint ../output_dir/text_clean_baseline/RCLMuFN/model.pt \
    --text_name text_clean \
    --splits test \
    --device 0
```

### 2. 使用不同 Batch Size（显存不足时）

```bash
python evaluate.py \
    --checkpoint ../output_dir/text_clean_baseline/RCLMuFN/model.pt \
    --text_name text_clean \
    --batch_size 16 \
    --device 0
```

### 3. 评估在不同数据集上的泛化性

```bash
# 在 text_clean 上训练的模型，在 text_final 上测试
python evaluate.py \
    --checkpoint ../output_dir/text_clean_baseline/RCLMuFN/model.pt \
    --text_name text_final \
    --device 0
```

### 4. 自定义对比分析

```bash
# 只对比特定实验
python compare_results.py \
    --eval_dir ./eval_results \
    --output_dir ./custom_comparison
```

---

## 📂 目录结构

```
RCL_TMD/
├── evaluate.py                 # 单个模型评估
├── batch_evaluate.sh           # 批量评估脚本
├── compare_results.py          # 结果对比分析
├── eval_results/               # 评估结果目录
│   ├── text_clean_baseline_text_clean_20260114_220000_results.json
│   ├── text_clean_baseline_text_clean_20260114_220000_report.txt
│   ├── text_clean_baseline_text_clean_20260114_220000_test_confusion_matrix.png
│   ├── text_clean_baseline_text_clean_20260114_220000_valid_confusion_matrix.png
│   ├── ... (其他实验的结果)
│   ├── comparison_report_20260114_220000.txt
│   ├── comparison_test_20260114_220000.csv
│   ├── comparison_valid_20260114_220000.csv
│   ├── comparison_test.png
│   └── comparison_valid.png
└── ../output_dir/              # 训练模型目录
    ├── text_clean_baseline/RCLMuFN/model.pt
    ├── text_clean_aggressive/RCLMuFN/model.pt
    ├── text_clean_conservative/RCLMuFN/model.pt
    └── text_clean_hard_neg/RCLMuFN/model.pt
```

---

## ⚠️ 常见问题

### Q1: 运行 batch_evaluate.sh 时显示 "No models found"

**A**: 检查以下几点:
1. 模型是否已训练完成
2. 模型保存路径是否正确（默认 `../output_dir/`）
3. 检查 `batch_evaluate.sh` 中的 `EXPERIMENTS` 数组是否包含你的实验名称

### Q2: evaluate.py 报错 "Checkpoint not found"

**A**: 确认 checkpoint 路径正确:
```bash
ls -lh ../output_dir/text_clean_baseline/RCLMuFN/model.pt
```

### Q3: 评估时 OOM (显存不足)

**A**: 减小 batch size:
```bash
python evaluate.py --checkpoint ... --batch_size 16
```

### Q4: 混淆矩阵图无法显示中文

**A**: 这是正常的，类别名称使用英文（Non-Hate, Hate）。

### Q5: 想要保存每个样本的预测结果

**A**: 使用 `--save_predictions` 参数:
```bash
python evaluate.py --checkpoint ... --save_predictions
```
然后查看 JSON 文件中的 `predictions` 和 `probabilities` 字段。

---

## 📊 推荐工作流

### 完整评估流程

```bash
# 1. 训练所有实验（假设已完成）
# bash train_single_stage.sh
# bash train_exp_aggressive.sh
# bash train_exp_conservative.sh
# bash train_exp_hard_neg.sh

# 2. 批量评估所有模型
bash batch_evaluate.sh

# 3. 生成对比报告
python compare_results.py --eval_dir ./eval_results

# 4. 查看结果
cat eval_results/comparison_report_*.txt

# 5. 可视化对比（在本地查看 PNG 文件）
# eval_results/comparison_test.png
# eval_results/comparison_valid.png
```

**预计总时间**: 20-40 分钟（取决于模型数量）

---

## 📝 输出文件清单

### 每个模型生成的文件（2 个 split × 2 个文件 + 1 个 JSON）

- `{experiment}_{dataset}_{timestamp}_results.json` - 详细指标（JSON）
- `{experiment}_{dataset}_{timestamp}_report.txt` - 可读报告（TXT）
- `{experiment}_{dataset}_{timestamp}_test_confusion_matrix.png` - 测试集混淆矩阵
- `{experiment}_{dataset}_{timestamp}_valid_confusion_matrix.png` - 验证集混淆矩阵

### 对比分析生成的文件

- `comparison_report_{timestamp}.txt` - 综合对比报告
- `comparison_test_{timestamp}.csv` - 测试集对比表格
- `comparison_valid_{timestamp}.csv` - 验证集对比表格
- `comparison_test.png` - 测试集可视化对比
- `comparison_valid.png` - 验证集可视化对比

---

## 🎯 下一步

评估完成后，根据结果：

1. **如果某个实验表现最好** → 使用该配置训练更多 seeds (42, 123, 456)
2. **如果所有实验都不理想** → 检查 WandB 日志，调整超参数
3. **如果需要提交结果** → 使用 `comparison_report_*.txt` 和 PNG 图

祝评估顺利！ 🚀
