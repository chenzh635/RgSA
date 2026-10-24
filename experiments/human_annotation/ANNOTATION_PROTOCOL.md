# Phishing Sentence Localization — Human Annotation Protocol (v2)

版本：v2（2026-08-31 修订）。v1→v2 变更：改用 `annotate.html` 盲标工具；新增仲裁规则、"?" 理由强制、签名/免责句默认 N 约定、良性邮件处理；标注者不再可见模型分与规则分。

## Task

For each email, identify which sentence(s) make it a phishing email. 每封邮件逐句标 **Y / N / ?**。

> **首次标注必读**：打开 `annotate.html` 的第一屏即为背景导读（3 分钟：任务说明、钓鱼邮件四段式结构、快判规则、易错点），点「我已了解」后才进入标注。

## Workflow（v2 流程）

1. 打开 `annotate.html`（浏览器，无需联网），填写你的名字，加载数据文件（pilot 阶段为 `pilot_30.json`）。
2. 逐封阅读邮件全部句子，用按钮或快捷键标注：`↑↓` 选句，`Y`/`N`/`?` 标注并自动下移，`←→` 换邮件。
3. 进度自动保存在本机浏览器。**标注完成后点「导出 CSV」与「导出 JSON」**，文件名形如 `annotations_<名字>_pilot_30_<时间>.csv`，发给项目负责人。
4. **盲化约定**：工具与数据文件均不含模型注意力分、规则分、模型预测——请仅依据邮件文本判断。不要查阅任何模型输出后再标注。

## What counts as a "phishing sentence"?

标 **Y**（钓鱼句）——句子意图满足以下任一条：
- Creates false urgency（制造紧迫感："24 小时内验证，否则账户冻结"）
- Impersonates an authority（冒充权威：伪装银行/公司/官方机构口吻）
- Requests sensitive information（索取敏感信息：账号、密码、卡号、身份证号）
- Contains deceptive links or instructions（诱导点击欺骗性链接、下载附件）
- Other social manipulation（其它社交操纵：冒充熟人、利益诱惑、恐吓）

标 **N**（非钓鱼）：
- 普通问候与客套（"Dear Customer" 单独出现且无恶意后续）
- **签名、免责声明、页脚、版权声明——一律 N**
- 中性陈述、无欺骗内容的说明

标 **?**（不确定）：**必须附一句理由**（工具强制，缺理由无法导出）。宁可标 ? 也不要瞎猜。

## 判断约定（v2 新增）

1. 看**意图**而非关键词；结合上下文——单看无害的句子在上下文中可能是钓鱼。
2. 一封邮件可以有**多句** Y，标出所有成立的。
3. **有的邮件完全不含钓鱼句**（此时全部标 N）。不要"必须找出点什么"。
4. 良性邮件中的孤立可疑句：若句子本身确有上述恶意意图才标 Y；仅因邮件整体可疑而句子本身无害的，标 N。
5. 链接行（URL）：若该链接服务于诱导行为（"点击此处验证"），与其引导句同标；纯引用/签名中的链接标 N。
6. **中文译文**（pilot 批次起提供）：每句附参考译文（原文在下方小字）。译文仅供理解，**判断以句子意图为准**；译文与原文冲突或词不达意时，以原文为准，拿不准标 ? 并说明。法语/乱码/代码片段已如实标注（如「【法语，多为乱码】」）。

## Adjudication（仲裁规则，v2 新增）

1. 所有标注者**独立**完成，标注期间不互相讨论。
2. 合并 gold 标签：3 人取多数表决；2 人对分歧句讨论达成共识。
3. 分歧句（无多数或僵持）集中讨论，记录于 `disagreements.csv`（可作论文 qualitative 素材）。
4. **一致性门槛**：pilot 阶段逐句 Cohen's κ（两两）/ Fleiss κ（≥3 人），κ ≥ 0.6 才进入全量标注；否则修订定义重做 pilot。

## Evaluation Metrics

标注将用于计算（对 7 个模型横向比较）：

1. **Precision@k**（k=1/3/5）：模型注意力 top-k 句中被标 Y 的比例（主指标）
2. **Recall@K**：人工标 Y 的句子落入模型 top-K 的比例
3. 句子级 **F1@k / IoU**、可用性曲线（达到给定 recall 需阅读的句数）
4. **Inter-annotator agreement**：Cohen's κ / Fleiss κ（按 stratum 分层报告）

边界处理：钓鱼邮件无人标出任何 Y 句 → 剔出 P@k 分母并单独计数；top-k 落 "?" 句按 N 计（另报剔除 ? 的敏感性版本）。

## Batches & Time Estimates

- **Pilot**：30 封（低相关 10 / 误分类 5 / 中 8 / 高 7），共 213 句，约 **30–45 分钟/人**
- 全量规模视 pilot 的 κ 与耗时决定（候选 100 或 150 封）
- 建议 2–3 名标注者（本次：3 人全量独立标注）

## 内部备忘（不发给标注者）

- `pilot_manifest.csv` 含各邮件 spearman/entropy——仅用于结果分层分析，**不得**向标注者展示。
- 数据生成器：`experiments/scripts/prepare_human_annotation.py`；pilot 抽样：`experiments/scripts/select_pilot.py`（seed=0）。
- 一致性脚本：`src/annotation_agreement.py`。
