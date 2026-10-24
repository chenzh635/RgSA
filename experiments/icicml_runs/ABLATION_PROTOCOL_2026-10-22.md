# 第二轮终评开封冻结协议（Frozen Protocol — Second Terminal-GT Opening）

- **协议日期**：2026-10-22（本文件落盘时间早于任何 GT 读取；以 git/文件时间戳为准）
- **作者批准**：2026-10-22 会话内书面批准（"批准，先写协议再跑"）
- **触发依据**：审稿单 `paper/submission/ICICML_2026/reviewer_report_2026-10-15.md` R1、R5、R6、R7 及其第四节响应规则第 3 条
- **服务稿件**：ICMLC 2027 LNNS 长稿（作者 2026-10-22 决定：整改只折入长稿，4 页短稿不再改动）
- **开封范围**：`experiments/agent_annotation/exports/gold_final.csv`，strict 模式（? 计 N），仅限 **202 个金标正袋**，评测器固定为 `src/evaluate_sentence_localization.py`（与首轮终评同一代码路径），配对 bootstrap B=10,000。

## 0. 不可违反的规则

1. 本协议先落盘，再运行任何读取 GT 的脚本。
2. 全部预先指定的配置一律报告，**不得**在看到 GT 结果后增删配置、更换种子、重选 epoch/λ、调整括号定义。
3. 训练期模型选择仅允许既有规则：固定 2,000 封验证集 AUPRC（accuracy 破并列），与首轮完全一致；GT 只在所有权重落盘后一次性评分。
4. 任何复现/对齐检查失败（数值超差、文件缺失、计数不符）必须原样写入输出 JSON 的 `parity_checks` 字段并在论文中如实报告，不得修补后静默重跑。
5. 本批次为该 GT 的**最后一次**开封；后续任何新想法需另立协议并重新取得作者批准。

## A. R1 — BERT 侧机制消融（需训练，缓存向量头训练）

目的：把完整模型相对 BERT-ABMIL 的捆绑总效应分解为规则项、稀疏项、退火（含推理温度，见 A.3）的贡献。

A.1 配置（f=1.0，即 30,000 训练袋；种子 123/456/789；10 epochs；其余超参数与 `icicml_run_cached.py` 文档串完全一致）：

| variant | λ_rule | λ_sparse | 训练 τ | 推理 τ |
|---|---|---|---|---|
| `full`（对齐参照，非新结果） | 1.0 | 0.1 | 1.0·0.95^epoch，floor 0.1 | 0.1 |
| `rule_only` | **1.0** | **0** | 同 full 退火 | 0.1 |
| `sparse_only` | **0** | 0.1 | 同 full 退火 | 0.1 |
| `no_annealing` | 1.0 | 0.1 | **固定 1.0** | **1.0** |

- λ_rule=1.0 是 f=1.0 下首轮机械选定值（验证集，非 GT）；rule_only/sparse_only 不重新选 λ。
- `no_annealing` 的推理 τ 预指定为 1.0（与其训练温度一致）；该变体因此隔离的是"退火训练+低温推理"的联合贡献（c)+(d)）。不设次级变体、不做事后拆分。
- 脚本：新建 `experiments/scripts/icicml_ablation.py`，复用 `icicml_run_cached.py` 的数据/模型/评测组件；输出 `experiments/icicml_runs/final_ablation_{variant}_seed{seed}.json`（含全测试集 `weights`，命名延续 final_* 规范）。

A.2 开封前对齐门（只看验证/测试邮件指标与训练历史，**不读 GT**）：
- 新脚本 `full` 变体与既有 `final_f1_sparse_seed{123,456,789}.json` 的 best_epoch、val AUPRC、test AUPRC/F1、entropy 逐项比较，容差 ≤0.002（MPS 非确定性）；超差即触发规则 4。
- 三个新变体 9 个训练全部落盘后，才允许运行 GT 评分。

A.3 GT 指标：每变体 3 种子的 P@1/P@3/R@3/k@90/H/email F1 均值±std；完整模型减每变体的配对 per-email bootstrap CI（种子平均口径，与首轮相同）。输出追加至 `experiments/icicml_runs/ablation_gt.json`。
论文用词预约定：规则项关闭后 P@1 退化最大 → 机制称谓以 "rule-guided" 为主；稀疏项独立贡献为负 → 不称 "sparse" 为独立致效成分（长稿相关结论据此改写，强弱以 CI 是否排零为准）。

## B. R5 — 固定 λ 跨分数对照（免训练，复用既有权重）

既有 `sweep_f{0_1,0_25,0_5,1}_lam{1,2,3,3_5}_seed42.json`（16 个，seed 42，已确认含全测试集 weights；开封前核验 n_train = 3,000/7,500/15,000/30,000）。

- 主对照：λ 固定 1.0 时 f∈{.1,.25,.5,1} 的 P@1/R@3/email F1，给出"λ 不变时标签分数的边际效应"；
- 描述性附表：全部 4 λ × 4 分数 P@1 矩阵（seed 42，仅描述，不做选择、不做多重比较宣称）；
- 与首轮 3-seed 机械选 λ 曲线（`gt_final.json`）并列，用于界定 f=0.25 的 P@1 峰值有多少来自 λ=3 的牵引；
- 输出 `fixed_lambda_gt.json`。不重训任何模型。

## C. R6 — 随机锚点（只读 gold，不涉及模型）

脚本 `experiments/scripts/icicml_random_anchor.py`，固定随机种子 **20261022**：
- 解析期望：逐袋超几何——P@k 期望为 top-k 至少含 1 个金标句的概率 1−C(n−k,g)/C(n,g)，对 202 袋取均值；R@k 期望为 min(k,g)/n 的均值；k@90 期望同法按定义计算；
- 蒙特卡洛：每袋均匀随机排列 **10,000 次**（≥审稿要求的 1,000），报告均值与 2.5/97.5 百分位；
- 解析与 MC 互证容差 0.002；审稿单给定参照值 P@1=P@3=P@5≈0.439、R@3≈0.579、R@5≈0.771；
- 输出 `random_anchor.json`（含金标集计数：袋数、金标句数、正袋总句数，与首轮记录 202/539/1,650 核对）。
长稿 Table（paper.tex:228）的 0.490 单次抽签行替换为该锚点，:245 的 CI 叙述同步重算。

## D. R7 — 56 袋括号（免重训，复用 final_* weights）

脚本 `experiments/scripts/icicml_sparse_brackets.py`：
- 分数：Sparse BERT 与 BERT-ABMIL 各 3 种子（`final_f1_{sparse,plain}_seed*.json`）逐袋种子平均权重；
- 括号定义复用 `experiments/scripts/m4_sensitivity.py:variant_labels` 与既有判定 `experiments/m4_review/m4_verdicts.json`（14 a / 5 b / 37 c），不得重新判定：
  - `baseline`（strict，202 袋）、`mixed`（主重裁定）、`all-c`（全部按漏标处理，审稿单最低要求）；
  - `all-a`、`all-b` 一并计算入 JSON（不额外宣称，仅供存档）；
- 指标 P@1/R@3；每括号每模型；输出 `sparse_brackets_gt.json`；
- 交叉核验：同一括号机器上 CNN 家族数字与 `experiments/m4_sensitivity.json` 一致（一致才允许写稿引用）。

## E. 执行顺序与产物清单

1. 本协议落盘（已完成）。
2. 开封前门检（不读 GT）：sweep n_train 计数、m4 判定计数、`full` 变体对齐门。
3. 训练 A（9 个消融头，约 15 分钟，MPS）。
4. 一次性运行 GT 评分：C → D → B → A（先纯计算后训练产物），输出 4 个 JSON。
5. 数字映射表追加到 `review_response_2026-10-15.md`，再改长稿、重编译、全新上下文子 agent 复核。

产物（均在 `experiments/icicml_runs/`，脚本在 `experiments/scripts/`）：
`icicml_ablation.py`、`icicml_random_anchor.py`、`icicml_sparse_brackets.py`、`icicml_round2_gt.py`（A/B/D 评分合一，带 final_* 齐全守卫，仿 `icicml_gt_final.py`）；
`final_ablation_{rule_only,sparse_only,no_annealing}_seed{123,456,789}.json`、
`ablation_gt.json`、`fixed_lambda_gt.json`、`random_anchor.json`、`sparse_brackets_gt.json`。
