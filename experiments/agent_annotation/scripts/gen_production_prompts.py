"""Generate 33 ready-to-paste production prompts (3 passes x 11 chunks)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # experiments/agent_annotation
template = (ROOT / "prompts" / "annotator_prompt.md").read_text(encoding="utf-8")

# Agent-facing body = everything after the first horizontal rule
body = template.split("\n---\n", 1)[1].lstrip("\n")

# Sanity: template must contain the exact anchors we replace
anchors = [
    "`experiments/agent_annotation/tasks/pilot_30_blind.json` — 你要标注的邮件",
    "（启动前把本行的占位替换为本 pass 名，如 `agent_pass_a` / `agent_pass_b` /\n`agent_pass_c`；三个 pass 各用独立会话、独立 pass 名，不要复用。）\n",
    "`<PASS_NAME>` = agent_pass_c",
    "`experiments/agent_annotation/outputs/<PASS_NAME>_<batch名>.json`",
    "格式严格遵循协议 §7（含 confidence 字段；annotator 字段 = `<PASS_NAME>`）。",
]
for a in anchors:
    assert a in body, f"anchor missing: {a[:40]!r}"

outdir = ROOT / "prompts" / "production"
outdir.mkdir(exist_ok=True)

made = []
for p in ("a", "b", "c"):
    pass_name = f"agent_pass_{p}"
    for n in range(1, 12):
        text = body
        # 1) task file: pilot -> production chunk
        text = text.replace(
            "`experiments/agent_annotation/tasks/pilot_30_blind.json` — 你要标注的邮件",
            f"`experiments/agent_annotation/tasks/production_chunk_{n}.json` — 你要标注的邮件\n"
            f"   （本会话只标注这一个 chunk，不要读取其他 chunk 或 pilot 文件）",
        )
        # 2) drop the human-facing placeholder note in 输出 section
        text = text.replace(
            "（启动前把本行的占位替换为本 pass 名，如 `agent_pass_a` / `agent_pass_b` /\n"
            "`agent_pass_c`；三个 pass 各用独立会话、独立 pass 名，不要复用。）\n",
            "",
        )
        # 3) fix pass name
        text = text.replace("`<PASS_NAME>` = agent_pass_c", f"`<PASS_NAME>` = {pass_name}")
        # 4) fix output path
        text = text.replace(
            "`experiments/agent_annotation/outputs/<PASS_NAME>_<batch名>.json`",
            f"`experiments/agent_annotation/outputs/{pass_name}_production_chunk_{n}.json`",
        )
        # 5) fix annotator field
        text = text.replace(
            "格式严格遵循协议 §7（含 confidence 字段；annotator 字段 = `<PASS_NAME>`）。",
            f"格式严格遵循协议 §7（含 confidence 字段；annotator 字段 = `{pass_name}`）。",
        )
        # generated-file header (one line, agent-safe)
        header = (
            f"> 自动生成：production 标注 prompt — pass `{pass_name}` / chunk {n}（共 11）。\n"
            f"> 将下方全部内容作为会话首条指令粘贴；本会话只产出 chunk {n} 的标注文件。\n\n---\n\n"
        )
        out = outdir / f"pass_{p}_chunk_{n:02d}.md"
        out.write_text(header + text, encoding="utf-8")
        made.append(out.name)

print(f"generated {len(made)} files in {outdir}")

# Verify: no unresolved placeholders except the single `<PASS_NAME>` definition line
bad = []
for f in sorted(outdir.glob("*.md")):
    t = f.read_text(encoding="utf-8")
    p = f.name.split("_")[1]  # pass letter
    for tok in ("<TASK_FILE>", "pilot_30_blind", "<batch名>"):
        if tok in t:
            bad.append((f.name, tok))
    if t.count("<PASS_NAME>") != 1 or f"`<PASS_NAME>` = agent_pass_{p}" not in t:
        bad.append((f.name, "PASS_NAME definition"))
    exp_task = f"tasks/production_chunk_{int(f.name.split('_')[-1].split('.')[0])}.json"
    if exp_task not in t:
        bad.append((f.name, f"missing {exp_task}"))
    exp_out = f"outputs/agent_pass_{p}_production_chunk_{int(f.name.split('_')[-1].split('.')[0])}.json"
    if exp_out not in t:
        bad.append((f.name, f"missing {exp_out}"))
print("verification issues:", bad if bad else "NONE")
