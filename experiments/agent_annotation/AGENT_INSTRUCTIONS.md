# 标注 Agent 操作总控（Runbook v2）

本文件给**协调者**（用户或主 agent）使用：开 agent 的顺序、每步命令、门槛判读。
标注 pass 自身的指令见 `prompts/annotator_prompt.md`（粘贴给每个新开的标注 agent）。
设计依据：`review/groundtruth_annotation_spec_2026-09-12.md`（三铁律）+
人工协议 v2/v3。

## 总原则（三铁律的执行方式）

1. **裁判与模型不同源**：标注 agent 用通用前沿 LLM；3 个 pass 尽量用
   **不同厂商/模型**（如 Claude + GPT + Gemini）。只有单一模型可用时，
   3 个会话仍须完全独立，并在论文披露同质性局限。
2. **不给答案册**：agent 只读协议 + 盲化任务文件；prompt 已明令禁止读
   `kwords_stem.json` / `src/rules.py`；阶段 0 有自动泄露审计。
3. **真人校准**：pilot 30 封已有 chan/li 两名真人标注；agent 必须过
   κ 门槛才允许进 production。

其他纪律：
- **每个 pass = 一个全新会话**，互相不可见输出（模拟独立标注员）。
- 支持的话固定 temperature=0（可复现）；无法固定则披露。
- **先 pilot 后 production**；门槛不过不产 gold。

## 阶段 0：泄露审计（一次性，已完成可跳过）

```bash
python3 experiments/agent_annotation/scripts/check_prompt_leakage.py   # 须 PASS
```

任务文件由 `prepare_agent_tasks.py` + `sample_supplement.py` 生成（已运行，
含泄露断言）：pilot 30 + production 258 封（原 150 + 失败模式扩样 108：
mismatch 60 / short 40 / diffuse 40 / typical 118，见 `tasks/_sampling_report.json`）。
若修改协议/prompt 后须重跑审计。

## 阶段 1：Pilot 校准（30 封 ×3 pass）

对 pass ∈ {a, b, c} 各开一个新会话，粘贴 `prompts/annotator_prompt.md`
分隔线内内容，替换：

- `<PASS_NAME>` → `agent_pass_a` / `agent_pass_b` / `agent_pass_c`
- `<TASK_FILE>` → `experiments/agent_annotation/tasks/pilot_30_blind.json`

产出：`outputs/agent_pass_{a,b,c}_pilot_30_blind.json`

（可选，位置偏差诊断）用 pass a 同款模型再开一个会话标
`tasks/pilot_30_blind_shuffled.json`，`<PASS_NAME>` = `agent_pass_a_shuffled`。

三个 pass 齐后评分：

```bash
python3 experiments/agent_annotation/scripts/score_agent_calibration.py \
    experiments/agent_annotation/outputs/agent_pass_a_pilot_30_blind.json \
    experiments/agent_annotation/outputs/agent_pass_b_pilot_30_blind.json \
    experiments/agent_annotation/outputs/agent_pass_c_pilot_30_blind.json
```

（可选）位置偏差：

```bash
python3 experiments/agent_annotation/scripts/score_position_bias.py \
    experiments/agent_annotation/outputs/agent_pass_a_pilot_30_blind.json \
    experiments/agent_annotation/outputs/agent_pass_a_pilot_30_blind_shuffled.json
```

产出：`outputs/calibration_report.json`（κ 门槛 + 邮件级 Y 集吻合 + P@1 式
命中率 + oracle 覆盖率 + low-confidence 占比 + pilot 分歧清单）、
`outputs/position_bias_report.json`。

### 门槛判读

| 门槛 | 内容 | 硬阈值 | 目标 |
|---|---|---|---|
| G1 | 每 pass 对两真人的平均 κ | ≥0.50（轮1重定基线，见 CALIBRATION_LOG.md §B） | — |
| G1b | parity：对每位人类 h，κ(agent,h) ≥ κ(另一位人类,h) | ≥κ(chan,li)=0.398（动态重算） | — |
| G2 | pass 两两 κ | ≥0.60 | ≥0.7（任务书） |
| G3 | 每 pass 对人类平均 κ ≥ κ(chan,li) | ≥0.398（动态重算） | — |

> **基线说明**：G1 原定 0.60 出自任务书 §7，但人类互标 κ 仅 0.398（excl-4860
> 0.471），要求 agent 对同一噪声锚达 0.60 在统计上不可达。轮1据观察到的标注者
> 间天花板重定为 0.50，并新增 G1b parity（"agent 至少是一名合格的人类标注员"）。
> 论文如实披露此调整。

附排除 email 4860（人工分歧主源）敏感性版；邮件级指标作佐证
（P@1 式命中、Y 集 Jaccard、覆盖率——真人 pilot 覆盖率仅 0.63，
若 agent 覆盖率≈1.0 说明 agent 有"全 Y 倾向"，须在判读中扣分）。

**操作性门槛**：人类互标 κ 仅 0.398（excl-4860: 0.471）且锚自身有内部
不一致，G1=0.50 对该噪声锚偏高。判定以 **G1b parity（agent 对每位人类 ≥
另一位人类对他）+ G3（≥κ(chan,li)）为操作性门槛**，G2≥0.60（目标 0.7）为
稳定性门槛，G1=0.50 为理想目标（论文如实披露此层级）。

- **G1b+G3+G2 PASS（full）** → 阶段 2（即使 G1<0.50，只要 parity 达成即
  视为"合格人类标注员"，可进 production）。
- **FAIL** → 读 report 的 `pilot_disagreements` 与邮件级指标定位系统性偏差
  方向（从宽=多标 Y / 从紧=漏标 Y / 签名误标 / 关键词跟随），修订协议判例
  与 prompt → 重跑阶段 0 泄漏审计 → 重跑 3 pass。
  - 轮 1（v1.1）从宽 FAIL → 轮 2（v1.2）；轮 2 从紧过正 FAIL → 轮 3（v1.3）。
  - **收敛条件**：当一轮修订不再显著移动 agent~human κ、或残余分岐主要由
    人类锚内部不一致驱动（非 agent 可修）时，停止迭代 → 回退（README
    「失败回退」：人工 30 封 GT 进附录 + agent GT 进主表并双向披露，
    见 `review/response_2026-09-12.md` §7），或经用户决定人工仲裁重锚。
  - **轮 3 终局（2026-09-13，用户决定）**：单 pass 字面 FAIL（pass_a mean
    0.400 过 G3，agent~chan 0.388 差 parity 0.010），但**多数票集成**
    κ=0.381/0.380 vs 基线 0.398，bootstrap 95% CI [−0.12,+0.16] 跨零——
    与人类天花板统计不可区分，且覆盖率恰落在两人类之间（无过冲）。收敛
    条件触发（残余由人类锚内部不一致主导），不跑轮 4。**接受集成 parity，
    进入阶段 2 production**；论文披露口径见 CALIBRATION_LOG.md 轮 3 节。

## 阶段 2：Production（258 封 = 11 chunk ×3 pass）

**33 个即用 prompt 已生成**（2026-09-13，占位已全部替换并校验）：
`prompts/production/pass_{a,b,c}_chunk_{01..11}.md`。每个文件对应一个独立
会话：把文件全文作为会话首条指令粘贴即可，无需任何手动替换。

- 每 chunk 一个会话（避免上下文吃紧），pass a/b/c 之间互相独立、不沿用
  pilot 会话；三个 pass 尽量用不同厂商/模型（铁律 1）
- 每个会话产出 `outputs/agent_pass_X_production_chunk_N.json`（文件内已写明）
- 若 prompt 需重新生成：改 `prompts/annotator_prompt.md` 模板后重跑生成脚本
  （逻辑见 prompts/production/ 文件头，替换 task 路径、pass 名、输出路径、
  annotator 字段四处），并重跑阶段 0 泄漏审计

产出：每 pass 11 个文件，共 33 个。

注意：pilot 30 封 ⊂ production 前 150 封（已核实）。agent 会盲态重标它们——
用于**自一致性校验**（任务书 §6 去重校验）：合并后对比 pilot pass 与
production pass 在这 30 封上的翻转率（协调者用 score_position_bias.py
同款逻辑或直接对比两份输出）。

## 阶段 3：合并 + 仲裁

1. 多数票合并 + 分歧清单 + 逐裁判原始标签 jsonl：

```bash
python3 experiments/agent_annotation/scripts/merge_and_adjudicate.py \
    --pass-a "experiments/agent_annotation/outputs/agent_pass_a_production_chunk_*.json" \
    --pass-b "experiments/agent_annotation/outputs/agent_pass_b_production_chunk_*.json" \
    --pass-c "experiments/agent_annotation/outputs/agent_pass_c_production_chunk_*.json"
```

产出：`gold_majority.json`、`production_disagreements.csv`、
`groundtruth_oracle_labels.jsonl`（任务书 §11.1）。

2. 开**仲裁 agent**（新会话，最好用第 4 个模型），粘贴
   `prompts/adjudicator_prompt.md`，`<DISAGREEMENTS_FILE>` →
   `experiments/agent_annotation/outputs/production_disagreements.csv`。
   产出：`outputs/adjudication.json`。

3. 折入仲裁，生成最终 gold + 一致性报告：

```bash
python3 experiments/agent_annotation/scripts/merge_and_adjudicate.py \
    --apply-adjudication experiments/agent_annotation/outputs/adjudication.json
```

产出：**`outputs/gold_final.json`**（论文 GT）、`annotation_agreement_report.md`
（任务书 §11.3，含待手填的 κ 汇总栏）。

## 阶段 4：GT 评测（协调者本机执行）

注意：部分模型检查点已于 2026-07 清理，运行前确认 `experiments/<model>/`
权重存在（备份见 `experiments/archive/`）；缺失的模型从 `--models` 剔除。

```bash
python3 experiments/scripts/run_gt_evaluation.py \
    --gold experiments/agent_annotation/outputs/gold_final.json \
    --models sparse,abmil,bert,sparse_bert,ai2tale,snorkel,rules,random \
    --mode strict --by-stratum --ks 1,3,5
# 敏感性：--mode exclude_q 再跑一遍
```

产出 `experiments/gt_localization_results.json`（P@k/R@k + bootstrap CI +
paired-vs-reference CI；random 基线已内置）。分层报告按失败模式
（mismatch/short/diffuse/typical）。

### 关键对照（任务书 §9，直接裁决"规则先验有没有价值"）

看 `results['ABMIL (CNN) [seedX]']['paired_vs_reference']['p@1']` 的
bootstrap CI（reference = Sparse ABMIL）：

- CI 不含 0 且方向为正 → 规则先验价值被证实，主表叙事照旧；
- CI 含 0 → 规则先验对"真找到"无显著贡献 → 论文主张改写
  （response 文件 §7 两手预案启动）。

## 盲态自查（每 pass 完成后协调者抽查）

```bash
# reason 中不得出现模型/分层词汇（出现即泄露，作废重跑）
grep -il "spearman\|stratum\|entropy\|attention\|rule.score\|quartile" \
    experiments/agent_annotation/outputs/agent_pass_*.json
```

覆盖数校验由 merge 脚本强制执行（缺句/重复/标签非法都会硬报错）。
