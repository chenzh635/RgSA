# 智能体集成标注 · 一致性报告（ensemble part）

- 最终 gold：1955 句 / 258 封（`gold_final.json`）
- 标签分布：{'N': 1390, 'Y': 539, '?': 26}（Y 占 0.276）
- confidence 分布：{'high': 1330, 'low': 123, 'medium': 501, 'unspecified': 1}（low 占 0.063）
- oracle 覆盖率（≥1 Y 的邮件占比）：202/258 = 0.783
- 仲裁折入：1 句（无多数票项）
- 句子数按分层：{'typical': 800, 'mismatch': 267, 'diffuse': 808, 'short': 80}

裁判间 κ、agent-vs-human κ、位置偏差结论见 `calibration_report.json` 与 `position_bias_report.json`
（pilot 阶段产出；数值应同步摘录到本文件下方）。

## 待协调者手填（跑完 pilot 校准后）

- 裁判间 Fleiss κ（3 agent）：＿＿＿
- agent-vs-human 平均 κ（G1）：＿＿＿
- κ(chan,li) 基线：0.398（excl-4860：0.471）
- 位置偏差（顺序 vs 打乱翻转率）：＿＿＿
- 裁判模型/厂商：＿＿＿（异质性披露）
