# YOLO11-seg 消融实验完整指南

## 你的任务背景
- **任务**：钢面缺陷实例分割（segmentation）
- **数据**：Severstal Steel Defect，图像长边 800px，矩形图较多
- **当前模型**：YOLO11s-seg（你之前训的是 detect，现在改 seg）
- **训练框架**：Ultralytics

## 消融实验路线图

```
Step 1: Baseline        → 原始 YOLO11n/s-seg（不做任何改动，只训练）
Step 2: +P2             → 增加 P2 小目标分割分支
Step 3: +P2 + EMA       → 在 P2 分支基础上，加入 EMA 轻量注意力
Step 4: +P2+EMA+BL      → 在前基础上，加入 Boundary Loss 边界损失
Step 5: +数据增强        → 在前基础上，加入强反光/模糊/低照度增强
```

每一步都单独训练，分别记录：
- **mask mAP50**、**mask mAP50-95**
- **Precision(M)**、**Recall(M)**
- **参数量**、**GFLOPs**、**FPS/推理时间**

---

## 目录结构

```
yolov11/
├── ablation/
│   ├── models/
│   │   ├── yolo11n-seg-p2.yaml        ← P2 分支模型配置
│   │   ├── yolo11s-seg-p2.yaml        ← P2 分支模型配置(s版)
│   │   ├── yolo11n-seg-p2-ema.yaml    ← P2+EMA 模型配置
│   │   ├── yolo11s-seg-p2-ema.yaml    ← P2+EMA 模型配置(s版)
│   ├── modules/
│   │   └── ema_attention.py           ← EMA 注意力模块
│   ├── losses/
│   │   └── boundary_loss.py           ← 边界损失实现
│   ├── augment/
│   │   └── strong_augment.py          ← 数据增强模块
│   ├── train_step1_baseline.py        ← Step1 训练脚本
│   ├── train_step2_p2.py              ← Step2 训练脚本
│   ├── train_step3_p2_ema.py          ← Step3 训练脚本
│   ├── train_step4_p2_ema_bl.py       ← Step4 训练脚本
│   ├── train_step5_augment.py         ← Step5 训练脚本
│   └── evaluate_all.py               ← 评估全部模型的脚本
```

---

## 前置知识（小白必读）

### 1. YOLO 多尺度检测原理
YOLO 有三个检测头：P3(8倍下采样)、P4(16倍)、P5(32倍)。
- P3 检测小目标（原图 8x8 以上）
- P4 检测中目标（原图 16x16 以上）
- P5 检测大目标（原图 32x32 以上）

**P2 是什么？** P2 是 4 倍下采样的特征图，分辨率是 P3 的 2 倍。
加上 P2 后，模型能检测更小的目标（原图 4x4 像素以上的缺陷）。

### 2. EMA（Efficient Multi-scale Attention）
一种轻量级的通道注意力机制。参数量极小，但能让模型更关注重要的
特征通道（比如缺陷边缘的纹理变化）。

### 3. Boundary Loss（边界损失）
普通的分割损失只看像素分类对不对。边界损失额外要求模型把缺陷的
边界轮廓也分清楚，对钢面裂纹这类细长缺陷特别有用。

### 4. 数据增强
训练时对输入图像做随机变换，让模型见过更多样的场景。
- 强反光 = 模拟车间照明变化
- 模糊 = 模拟相机失焦/抖动
- 低照度 = 模拟光线不足

---

## 目录结构

```
yolov11/
├── ablation/
│   ├── README.md                  ← 你正在看的文件（完整指南）
│   ├── models/
│   │   └── yolo11-seg-p2.yaml     ← P2 分支模型配置
│   ├── modules/
│   │   ├── ema_attention.py       ← EMA 注意力模块
│   │   └── boundary_loss.py       ← 边界损失实现
│   ├── augment/
│   │   └── strong_augment.py      ← 数据增强模块
│   ├── train_step1_baseline.py    ← Step1 训练脚本
│   ├── train_step2_p2.py          ← Step2 训练脚本
│   ├── train_step3_p2_ema.py      ← Step3 训练脚本
│   ├── train_step4_p2_ema_bl.py   ← Step4 训练脚本
│   ├── train_step5_augment.py     ← Step5 训练脚本
│   ├── run_all.py                 ← 一键运行 1~5 步
│   └── evaluate_all.py            ← 汇总评估（生成论文用对比表）
├── runs/
│   └── segment/
│       └── yolo.yaml/
│           ├── yolo11n.yaml       ← 下载的官方配置
│           └── yolo11n.yaml       ← 下载的官方配置
```

---

## 怎么跑（小白版教程）

### 第 1 步：确认环境
打开终端（cmd 或 PowerShell），切换到 yolov11 目录：
```
cd D:\projects\graduation_paper\yolo\yolov11
```
确认 Python 能看到 ultralytics：
```
python -c "import ultralytics; print(ultralytics.__version__)"
```

### 第 2 步：单独跑 Baseline（验证环境）
```
cd ablation
python train_step1_baseline.py
```
如果能正常训练，说明环境 OK。按 Ctrl+C 可以中断。

### 第 3 步：依次跑每一步
```
python train_step1_baseline.py    # Step 1: Baseline
python train_step2_p2.py          # Step 2: +P2
python train_step3_p2_ema.py      # Step 3: +P2+EMA
python train_step4_p2_ema_bl.py   # Step 4: +P2+EMA+BL
python train_step5_augment.py     # Step 5: +数据增强
```

或一键全部运行：
```
python run_all.py
```

### 第 4 步：汇总评估
```
python evaluate_all.py
```
会输出 Markdown 格式的对比表，直接复制到论文里即可。

---

## 每步怎么改代码？（小白版详解）

### Step 1 — Baseline（不改任何代码）
直接用 `model = YOLO("yolo11s-seg.pt")` 训练，不碰任何内部代码。
这一步是基准线，后续所有改进都和它比。

### Step 2 — +P2（改模型配置文件）
**改了什么？** `yolo11-seg-p2.yaml` 这个文件。

**怎么改的？**
```
原始 head:
  (P3/8) → (P4/16) → (P5/32) → 3 个检测头

P2 版 head:
  (P2/4) → (P3/8) → (P4/16) → (P5/32) → 4 个检测头
```
具体就是在 head 里多加了两次上采样和下采样（P3→P2→P3），
最后 Segment 头多收一个 P2 的特征。

**训练脚本改了什么？**
```python
# 原来：
model = YOLO("yolo11s-seg.pt")

# 改后：
model = YOLO("models/yolo11-seg-p2.yaml").load("yolo11s-seg.pt")
```
用自定义 YAML 构建模型结构，但加载预训练权重初始化。

### Step 3 — +P2+EMA（替换一个模块）
**改了什么？** 把 backbone 里的 C2PSA 模块替换为 C2PSA_EMA。

C2PSA 是 YOLO11 的注意力模块（用 PSA 做自注意力），
C2PSA_EMA 是我们自己写的版本（用 EMA 替换 PSA）。

**怎么改的（最简单的方式）：**
```python
import ultralytics.nn.modules as ult_modules
from ema_attention import C2PSA_EMA

# 全局替换！
ult_modules.C2PSA = C2PSA_EMA
```
一行代码搞定。后续模型加载 `yolo11-seg-p2.yaml` 时，
里面的 C2PSA 会自动变成 C2PSA_EMA。

### Step 4 — +P2+EMA+BL（注入边界损失）
**改了什么？** 在 Step 3 基础上，训练时的 loss 计算多了一项。

**怎么改的：**
```python
from ultralytics.utils.loss import v8SegmentationLoss

# 保存原始 forward
original_forward = v8SegmentationLoss.forward

# 写一个新的 forward
def patched_forward(self, preds, batch):
    loss = original_forward(self, preds, batch)
    # 加上 boundary loss
    loss = loss + boundary_loss_fn(preds, batch)
    return loss

# 注入
v8SegmentationLoss.forward = patched_forward
```

### Step 5 — +数据增强（调大增强参数）
**改了什么？** 训练时的数据增强参数。

**怎么改的：** 完全不需要修改 Ultralytics 源码，只需要把
`model.train()` 里的 hsv 参数、degrees、translate 等
调大。Ultralytics 本身就支持所有这些参数，默认值比较
保守，我们只是加大力度。

---

## 常见问题

**Q: 训练时报 "C2PSA_EMA 找不到"？**
A: 检查运行目录是不是在 `ablation/` 下面。
脚本开头的 `sys.path.insert` 把 `modules/` 加到了搜索路径。

**Q: Boundary Loss 没有生效？**
A: Step 4/5 的脚本用了 monkey patch，需要确保在 `import YOLO` 之前完成注入。
脚本已经帮你按正确顺序写了，直接运行即可。

**Q: 显存不够？**
A: 把 `batch=0.70` 改成 `batch=0.40` 或 `batch=16`。

**Q: 每步训练多久？**
A: 取决于 GPU。以 RTX 3060 为例，200 epochs × 800px ≈ 2~4 小时/步。
加上 P2 分支后会更慢（多了检测头），但差别不大。

**Q: 怎么选 n 还是 s？**
A: 推荐用 `s`（yolo11s-seg.pt），因为：
- P2 分支会略微减速，s 版本的速度损失比例更小
- s 的 Baseline mAP 更高，改进对比更明显
- 想省时间可以先跑 n，快速验证流程，最后用 s 出正式结果

---

## 消融对比表模板
训练完所有步骤后，运行 evaluate_all.py 会自动生成下表：

| 实验 | 说明 | mask mAP50 | mask mAP50-95 | Precision | Recall | 参数量(M) | GFLOPs | FPS |
|------|------|-----------|--------------|-----------|--------|----------|--------|-----|
| Step 1 | Baseline YOLO11s-seg | — | — | — | — | — | — | — |
| Step 2 | +P2 小目标分支 | — | — | — | — | — | — | — |
| Step 3 | +P2 + EMA 注意力 | — | — | — | — | — | — | — |
| Step 4 | +P2+EMA + Boundary Loss | — | — | — | — | — | — | — |
| Step 5 | +数据增强 | — | — | — | — | — | — | — |
