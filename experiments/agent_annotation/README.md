# LLM 智能体句级标注系统（agent_annotation）· v2

用 3 个独立盲态 LLM 标注 pass + 仲裁，替代昂贵的人工句级标注，
产出论文 GT（`gold_final.json`），直接接入 `run_gt_evaluation.py --gold`。
设计依据：`review/groundtruth_annotation_spec_2026-09-12.md`（三铁律）+
人工协议 v2/v3 + 已有 chan/li pilot 校准锚。

**本系统服务于论文对审阅意见 S2/S3/B1 的根治**：GT 落地后主表从循环的
Spearman-vs-规则换为 P@1/P@3/R@3（非循环），并直接裁决"规则先验训练有没有
价值"（任务书 §9 对照实验）。

## 三铁律 → 本系统的实现

| 铁律 | 实现 | 审计手段 |
|---|---|---|
| 1. 裁判与模型不同源 | 标注 agent = 通用前沿 LLM；3 pass 尽量异厂商；与被测 CNN/DistilBERT 编码器零交集 | runbook 要求披露裁判模型 |
| 2. 不给关键词答案册 | 协议只有行为定义、无字面触发短语；prompt 明令禁读 `kwords_stem.json`/`src/rules.py`；任务文件剥离全部模型字段 | `check_prompt_leakage.py`（类别名/触发短语/词干密度三重检查）+ 盲化断言 |
| 3. 真人校准 | chan/li ×30 封真人 pilot 作锚；κ 门槛不过不产 gold | `score_agent_calibration.py`（G1≥0.50 / G1b parity / G2 / G3 + excl-4860 敏感性；门槛轮1据人类天花板重定基线，见 CALIBRATION_LOG.md §B） |

## 相对 v1 的升级（对照任务书补齐的缺口）

1. **扩样到 258 封**（任务书 §2）：原 150 + 失败模式扩样 108
   （mismatch/Spearman<0 补到 60、short/2句 补到 40、diffuse/熵>1.5长邮件
   补到 40），分层指标直接对应论文失败模式叙事；短邮件取"恰好 2 句"
   （1 句邮件对 P@1 退化，与原 150 及 Spearman 排除规则同口径）。
2. **confidence 字段**（§3/§6）：Y/N/? + high/medium/low，
   不允许把不确定性强制二值化后丢弃；low 占比单列统计。
3. **位置偏差诊断**（§6）：`pilot_30_blind_shuffled.json` +
   `score_position_bias.py`（翻转率 + κ + 翻转方向混淆阵）。
4. **异质多裁判**（§6）：3 pass 建议不同厂商模型；不可得时披露同质局限。
5. **校准指标扩充**（§7）：κ 之外新增邮件级 Y 集 Jaccard、first-Y 一致、
   P@1 式命中率（agent 首个 Y ∈ 真人 Y 集）、oracle 覆盖率、
   low-confidence 占比；G2 目标 0.7（硬门槛 0.60）。
6. **自一致性校验**（§6 去重）：pilot 30 ⊂ production，agent 盲态重标，
   跨批次翻转率即自一致性。
7. **产出物对齐任务书 §11**：`groundtruth_oracle_labels.jsonl`（逐裁判原始
   标签+confidence+最终标签）、`calibration_report.json`（≈
   calibration_human_vs_llm）、`annotation_agreement_report.md`、
   `gt_localization_results.json`（含 bootstrap CI，Random 基线已补进
   `run_gt_evaluation.py`）。
8. **可复现性**（§4）：建议 temperature=0；采样扩样 seed=0；
   随机基线 seed=0。

## 目录结构

```
agent_annotation/
├── README.md                      ← 本文件（给协调者/用户）
├── AGENT_INSTRUCTIONS.md          ← Runbook v2：阶段 0–4 全部命令与判读
├── protocol/PROTOCOL_AGENT_v1.md  ← 协议 v1.3（agent 唯一判定标准，无答案册；版本演进见 CALIBRATION_LOG.md）
├── prompts/annotator_prompt.md    ← 标注 pass 启动指令（含盲态规则）
├── prompts/adjudicator_prompt.md  ← 仲裁 agent 指令
├── scripts/
│   ├── prepare_agent_tasks.py     ← 盲化任务生成（含 shuffled 变体）✅已运行
│   ├── extract_email_pool.py      ← 全测试集逐邮件指标池（sidecar）✅已运行
│   ├── sample_supplement.py       ← 失败模式扩样（seed 0）✅已运行
│   ├── check_prompt_leakage.py    ← 铁律 2 审计 ✅PASS
│   ├── score_agent_calibration.py ← pilot 校准 + 门槛 + 邮件级指标
│   ├── score_position_bias.py     ← 顺序 vs 打乱翻转率
│   └── merge_and_adjudicate.py    ← 多数票 / 仲裁折入 / jsonl / md 报告
├── tasks/                         ← 盲化任务（给 agent）
│   ├── pilot_30_blind.json / pilot_30_blind_shuffled.json
│   ├── production_chunk_1..11.json（258 封）
│   └── _key_stratum.json / _email_pool_metrics.json / _sampling_report.json
│       ← 旁挂 sidecar（含模型信息），⚠️ 绝不给标注 agent
└── outputs/                       ← 标注产物（现为空，待运行）
```

## 快速开始

按 `AGENT_INSTRUCTIONS.md` 阶段 0→4 顺序执行。核心节奏：

1. 3 个新会话标 pilot（尽量 3 个不同模型）→ 校准评分 → **PASS 才继续**
2. 3 个全新会话标 production 11 chunks
3. 合并 → 仲裁 agent → `gold_final.json`
4. `run_gt_evaluation.py --gold ... --by-stratum`（strict + exclude_q）→
   看 §9 对照（Sparse vs ABMIL 的 P@1 paired CI）→ 定论文叙事分支

预计开销：pilot 3×30 封很小；production 3×258 封为大头（每 pass 约 11 个
chunk 的顺序会话）；仲裁仅覆盖无多数句。

## 论文披露条款（写作时用，任务书 §10）

- gold 构成：3 独立盲态 LLM pass 多数票 + 第 4 agent 仲裁；裁判模型/厂商如实列出；
- 校准：agent-vs-human κ 与 human-vs-human κ（chan/li=0.398）并列报告；
  排除高分歧邮件的敏感性版；邮件级 P@1 吻合与覆盖率；
- 位置偏差与自一致性检查结果（翻转率）；
- 盲化字段清单 + 泄露审计脚本；
- 局限：LLM 是人类判断的代理，残余系统偏差无法完全排除；
  pilot-scale（258 封、单数据集、单语种）；P@1 给 bootstrap CI；
  真人锚仅 30 封 ×2 人。
- **干净测试集声明**：GT 邮件全部来自 test split，从未参与训练；
  oracle 标签不回流任何训练/调参。如实注明历史检查点按 test acc 选点
  （弱选择偏差），GT 指标本身不用于模型选择。

## 失败回退

- pilot 两轮修订后仍 FAIL → 放弃 agent gold：30 封真人 GT 进附录
  （统计功效有限，Limitations 写明）+ LLM 标注降级为辅助证据（仅报 κ）。
- 某 pass 中途崩溃 → 只重跑该 pass（新会话）。
- 覆盖率不一致 → merge 脚本硬报错列出缺失项，修补对应 chunk 再合并。
- 覆盖率异常（agent 近 100% vs 真人 63%）→ 提示 agent"全 Y 倾向"，
  按协议 §2（全 N 合法）修订 prompt 后重跑。
