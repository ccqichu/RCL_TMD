````markdown
# 请直接实现：Border-only/Center-only 与 Edge-mask/Crop 曲线（Test-time 变换评测）

你要做的是：在**不改训练**的前提下，通过 **test-time 图像输入变换**实现两组验证实验，并输出 CSV + 图表，便于我直接写论文。

---

## 0) 总目标与输出

### 实验 A：Border-only vs Center-only（k=10%）
对测试集每个样本（**文本保持不变**），评测三种图像输入：
- `original`：原图
- `border_only(k=0.10)`：**保留边缘环形区域**，中心区域用“中性填充/强模糊”替换
- `center_only(k=0.10)`：**保留中心矩形区域**，边缘环形区域用“中性填充/强模糊”替换

**输出：**
- `results/border_center_table.csv`
- 其中列至少包含：`model, setting, k, acc, macro_f1`（以及我项目已有的其他主指标）
- 并在 CSV 中额外输出：`delta_border = score(original)-score(border_only)`、`delta_center = score(original)-score(center_only)`（至少对主指标）

### 实验 B：Edge-mask / Crop 曲线（k 序列）
对测试集每个样本生成多档强度的变换，绘制性能随 k 变化曲线：

- `edge_mask(k)`：将**边缘环形区域**用“中性填充/强模糊”替换（中心不变）
- `crop_resize(k)`：裁掉 k% 边缘后 resize 回模型输入尺寸

推荐 k 序列（主文够用）：
- `k ∈ {0, 0.02, 0.05, 0.08, 0.10, 0.15}`  （k=0 表示原图）

**输出：**
- `results/mask_curve.csv`
- `results/crop_curve.csv`
- `results/mask_curve_macro_f1.png`（至少画 macro_f1，一张图一条曲线/或多模型多曲线）
- `results/crop_curve_macro_f1.png`
- 同时在 CSV 里输出：`delta10 = score(k=0)-score(k=0.10)`（对 mask/crop 各算一个，至少主指标）

---

## 1) 必须遵守的约束（不要违反）

1. **训练完全不改**：所有模型都按原始 train set 正常训练，不加入这些变换。
2. **变换只在 test-time 做**：在 eval loop 里，对图像 batch 做变换后再送入模型。
3. **文本不变**：只改图像输入；文本输入保持原样。
4. **对所有模型一致**：image-only、多模态 baseline、我的模型 CID–DIMM 都用同一套变换函数。
5. **复现性**：固定插值方式、固定 fill 策略、固定 k 序列；写在代码注释/README 里。

---

## 2) 需要新增的文件与功能

### 2.1 新增：`transforms_border.py`
实现 4 个函数（至少支持 PIL 图像；若我项目输入是 Tensor，也请你适配或在 eval 前转回 PIL）：

- `border_only(img, k, fill_mode="per_image_mean", blur_sigma=None)`
- `center_only(img, k, fill_mode="per_image_mean", blur_sigma=None)`
- `edge_mask(img, k, fill_mode="per_image_mean", blur_sigma=None)`（语义上等同 center_only）
- `crop_resize(img, k, out_size, interpolation="bilinear")`

#### 区域定义（必须一致）
对 H×W：
- `bh = round(k * H)`
- `bw = round(k * W)`
边缘区域 `B_k`：
- top `[0:bh, :]` + bottom `[H-bh:H, :]` + left `[:, 0:bw]` + right `[:, W-bw:W]`
中心区域 `C_k`：
- `[bh:H-bh, bw:W-bw]`

#### 填充策略（fill_mode）
默认使用：`per_image_mean`（每张图像自身 RGB 均值作为填充值），避免引入新 cue。
可选支持：
- `gray`：固定灰色（127）
- `imagenet_mean`：固定 ImageNet 均值像素（123,117,104）

可选加分：支持 `blur_sigma=8.0` 时，用强高斯模糊替换被遮挡区域（用于附录敏感性检验）。

---

### 2.2 新增：`eval_border_robustness.py`
功能：
1) 加载测试集 dataloader（复用我项目现有 dataset/dataloader）
2) 加载指定模型 checkpoint（复用我项目现有模型加载方式）
3) 对每个模型，在不同 transform / k 下跑完整 test
4) 计算指标（至少 Acc、Macro-F1；如果我项目已有主指标，也一并输出）
5) 保存 CSV + 画图（PNG）

#### 运行方式（你可以设计 CLI 参数）
例如：
```bash
python eval_border_robustness.py \
  --models image_only,text_only,baseline_mm,cid_dimm \
  --ckpts path1,path2,path3,path4 \
  --k_list 0,0.02,0.05,0.08,0.10,0.15 \
  --fill_mode per_image_mean \
  --out_dir results/
````

---

## 3) 建议实现方式（最少改动、最稳）

### 3.1 变换应用位置

不要改 dataset 的 `__getitem__`。保持 dataset 返回原始样本：

* image（PIL 或 Tensor）
* text（原样）
* label

在 `eval_border_robustness.py` 的 eval loop 里对 `image` 做变换（对 batch 内每张图逐个处理即可）。

### 3.2 统一预测接口

在脚本内封装：

* `predict_logits(model, images, texts) -> logits`

要求适配：

* image-only：忽略 texts
* text-only：忽略 images
* multimodal：两者都用
  并保证 logits 形状统一：`[B, num_classes]`

---

## 4) 结果输出格式（严格按这个）

### 4.1 `results/border_center_table.csv`

列至少包含：

* `model, setting, k, acc, macro_f1`
  其中 `setting ∈ {original, border_only, center_only}` 且 `k=0.10`

### 4.2 `results/mask_curve.csv` 与 `results/crop_curve.csv`

列至少包含：

* `model, transform, k, acc, macro_f1`
  其中 `transform ∈ {edge_mask, crop}`

### 4.3 图像输出

至少输出：

* `results/mask_curve_macro_f1.png`
* `results/crop_curve_macro_f1.png`

绘图要求：

* x 轴是 k
* y 轴是 macro_f1
* 每个模型一条曲线（同图多曲线）
* 图中标注清晰 legend

---

## 5) 最小正确性自测（你必须做，避免变换写错）

请写一个简单测试（可放 `tests/test_border_transforms.py` 或在脚本里 `--dry_run`）验证：

1. mask/border/center 变换后图像尺寸不变
2. crop_resize 输出尺寸等于 out_size
3. border_only 的中心区域确实被替换（抽样像素对比）
4. center_only 的边缘区域确实被替换（抽样像素对比）

---

## 6) 交付时你必须说明

1. 你新增/修改了哪些文件
2. 如何运行脚本（完整命令）
3. 输出文件有哪些
4. 如果你无法对接我的工程，请明确列出你还需要我提供的最小信息（见下一节）

---


---

请按以上要求完成实现。


