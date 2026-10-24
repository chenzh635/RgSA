# M1 复现工作区（AI2TALE 原始参考实现）

**目的**：审阅意见 M1——验证 AI2TALE 作者参考实现在原始数据上能否正常工作（自审中本项目重写版 F1=0.33、P@1 低于随机）。

## 来源与版本

- 代码：`ai2tale_src/`（作者 GitHub 发布版，git 快照保留于该目录）
- 环境：`/Users/miafiles/.venvs/ai2tale_tf`（Python 3.9.6，`tensorflow==2.15.1`、`tensorflow-probability==0.23.0`；作者标注环境为 TF 2.9 / Py3.9，2.15 为 Apple Silicon arm64 上可用的最接近 Keras-2 API 版本）
- 数据：`phishingdata/` 原始 .mat（x_y_tr_m.mat / x_y_te_m.mat，键与 shape 已核对一致；X_*_len 为 0/1 mask）

## 对参考代码的补丁（最小化）

1. `ai2tale.py` GPU 初始化：Mac 无 CUDA 设备，`gpus[0]` 会 IndexError → 加 `if gpus:` 守卫。
2. 优化器：TF ≥2.11 的新 Keras Adam 在交替训练（两阶段变量子集不同）下抛 KeyError → 改用 `optimizers.legacy.Adam`（与作者 TF2.9 上的 Keras Adam 行为一致，官方迁移指引推荐）。
除此之外无其他改动。

## 运行

```bash
/Users/miafiles/.venvs/ai2tale_tf/bin/python experiments/m1_repro/ai2tale.py \
  --train --data phishingdata/ --home experiments/m1_repro/runs/ai2tale
```

默认超参（作者采样值）：tau=0.5, k=1, lr=0.001, epo=10, batch=128, sig=0.3, lam=0.1, slen=100, tlen=50, emb=150。
推理（测试集前半）：去掉 `--train`。注意作者代码把 8,000 测试邮件对开：valid 后半用于训练期验证、test 前半用于最终推理。

## 判定口径

- 原始路径能复现作者报告的 Label-Accuracy（高位）→ 问题在本项目集成/重写，随后做忠实 PyTorch 移植（硬选择 + 正确 IB 正则项）并 3 seeds 重跑 GT。
- 原始路径同样崩溃 → 记录为参考实现自身问题，论文中按 "unsuccessful reproduction attempt" 表述。
