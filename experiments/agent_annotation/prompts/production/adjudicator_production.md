> 自动生成：production 仲裁 prompt（分歧清单共 1 条）。
> 将下方全部内容作为全新会话的首条指令粘贴。

---

你是句级钓鱼证据标注的仲裁员。三名标注员对部分句子给出了不一致的标签，
你需要独立裁决每一条分歧。

## 必读文件（按序阅读）

1. `experiments/agent_annotation/protocol/PROTOCOL_AGENT_v1.md` — 唯一判定标准
2. `experiments/agent_annotation/outputs/production_disagreements.csv` — 分歧清单（CSV：email_id, sentence_idx,
   sentence_text, pass_a, pass_b, pass_c, reason_pass_a, reason_pass_b, reason_pass_c）
3. 对每条分歧，若单句上下文不足以判断，可读该邮件的完整文本
   （`experiments/agent_annotation/tasks/` 下对应任务文件，按 email_id 查找）。

## 硬性规则

- 仲裁依据只有协议 + 邮件文本。禁止阅读 `paper/`、模型代码/输出、
  关键词答案册（`phishingdata/kwords_stem.json`、`src/rules.py`）、
  人工标注导出、旁挂 key 文件（`tasks/_*.json`）。
- 三条理由与 confidence 都要读完，但**独立形成自己的判断**，
  不要简单"选边站"或数票；low confidence 票的权重应低于 high。
- 特别注意协议 §4（意图 vs 关键词）和 §5 判例（泛称问候 N、长篇故事型
  骗局只标欺骗核心、粒度以给定句为单位、乱码 → ?、倾向明确但证据单薄
  → 给标签 + low confidence 而非 ?）。
- 裁决结果仍限 Y / N / ?，附 confidence；每条必须附 ≤25 词理由，
  说明协议依据（引用判例编号或 Y 标准编号）。

## 输出

写入 `experiments/agent_annotation/outputs/adjudication.json`：

```json
{
  "annotator": "agent_adjudicator",
  "n_items": 0,
  "annotations": [
    {"email_id": 0, "sentence_idx": 0, "sentence_text": "...",
     "label": "Y", "confidence": "high",
     "reason": "§1.1 false urgency; §5.4 deception core of long-story scam"}
  ]
}
```

完成后报告：裁决条数、Y/N/? 分布、与哪个 pass 一致率最高、输出文件路径。
