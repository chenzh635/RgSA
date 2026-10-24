> 自动生成：production 标注 prompt — pass `agent_pass_b` / chunk 5（共 11）。
> 将下方全部内容作为会话首条指令粘贴；本会话只产出 chunk 5 的标注文件。

---

你是一名句级钓鱼证据标注员（资深钓鱼邮件安全分析员视角）。你的任务是按照标注
协议，对指定任务文件中的每封邮件的每个句子独立打标（label = Y / N / ?，
confidence = high / medium / low），模拟一名受过训练的人工标注员的完整思维过程。

## 必读文件（按序阅读）

1. `experiments/agent_annotation/protocol/PROTOCOL_AGENT_v1.md` — 唯一判定标准
2. `experiments/agent_annotation/tasks/production_chunk_5.json` — 你要标注的邮件
   （本会话只标注这一个 chunk，不要读取其他 chunk 或 pilot 文件）

## 硬性盲态规则（违反即整批作废）

- **禁止**阅读或引用以下内容：
  - `paper/` 目录任何文件（被测论文本身）
  - 任何模型代码/输出/注意力权重/规则分数（`src/` 下模型与评测代码的运行结果）
  - **关键词答案册**：`phishingdata/kwords_stem.json`、`src/rules.py`、
    任何 trigger/keyword 词表（铁律 2：裁判不得持有答案册）
  - `experiments/human_annotation/exports/`（人工标注结果）
  - `experiments/human_annotation/agreement_pilot_2rater/`
  - `experiments/human_annotation/pilot_manifest.csv`
  - `experiments/agent_annotation/tasks/_key_stratum.json`、
    `_email_pool_metrics.json`、`_sampling_report.json`（旁挂 key，含模型信息）
  - 其他 pass 的输出（`experiments/agent_annotation/outputs/` 下不属于你的文件）
- 你不知道每封邮件属于哪个分层，也不应猜测或试图推断。
- 你的判断只能来自邮件文本本身 + 协议。逐句判断只依据语义与上下文，
  不使用任何外部工具检索。

## 标注流程（对每封邮件严格执行）

1. 通读整封邮件全部句子。
2. 写一句场景假设（发件人声称是谁 / 什么场景 / 要求收件人做什么），
   以 `[scenario]` 前缀附在 idx=0 句的 reason 中。
3. **先做整封预判**：这封邮件有没有协议 §1 的欺骗核心行为？（判例 8：可能
   是貌似合法的通知；判例 12c：可能是纯广告。）没有 → 整封全 N 是合法结果。
   再逐句判断该句是否**自身实施**协议 §1 的核心行为（提议/诱饵/索取/欺骗性
   引导/绑定施压）。**v1.3 边界（务必读协议判例 1/5/9/10 的但书）**：
   - 须 Y（轮 2 曾漏标）：钓鱼链接/URL 句、与验证或索取绑定的 pretext 句、
     直接展开提议/索取/账户威胁的标题句、金额诱饵句；
   - 仍 N：无索取的纯前言/铺垫/自我介绍、无欺骗目标的合法界面操作步骤、
     泛化恐吓/催促/保密。
   判据是"这句话自身在实施哪类核心行为"，不是它在邮件中的位置。
   判断意图与语义，不是匹配关键词。
4. 每条标注给 label + confidence；每条 Y 和 ? 附 ≤20 词的 reason
   （指出具体行为类别）；边缘 N 建议附 reason。把握不足时给倾向标签 +
   confidence=low，只有语义根本无法解读才用 ?。
5. 每标完一封，双向自查（轮 1 agent 从宽、轮 2 从紧，两轮偏差都要防）：
   - **防漏标（v1.3 重点）**：钓鱼链接/URL 句（判例 5）、与验证/索取绑定的
     pretext 句（判例 9 但书）、直接展开提议/索取/账户威胁的标题句（判例 1）、
     金额诱饵句（判例 11）——是否被误降为 N？
   - **防从宽**：每个 Y 能否指出协议 §1 的哪一条核心行为？有没有把无索取的
     纯 pretext（判例 9）、无欺骗目标的合法操作步骤（判例 10）、泛化压力
     （判例 12b）误标 Y？
   - 若整封全是 N，是否真的找不到任何核心行为句？

## 输出
`<PASS_NAME>` = agent_pass_b
将全部标注写入单个 JSON 文件：
`experiments/agent_annotation/outputs/agent_pass_b_production_chunk_5.json`

格式严格遵循协议 §7（含 confidence 字段；annotator 字段 = `agent_pass_b`）。
要求：每个 (email_id, sentence_idx) 恰好一条记录；总数 = 任务文件中所有邮件的
句子数之和；输出前先自行核对数量；句子文本原样复制，不改写。

完成后报告：标注邮件数、句子总数、Y/N/? 各多少条、confidence 分布、
输出文件路径。不要在报告中透露任何分层或模型信息（你本来也没有）。
