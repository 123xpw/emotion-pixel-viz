import json
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
REQUIRED_FIELDS = [
    "primary_emotion",
    "secondary_emotions",
    "emotion_layers",
    "intensity",
    "valence",
    "arousal",
    "color_palette_cn",
    "scene_keywords_cn",
    "core_metaphor_cn",
    "supporting_metaphors_cn",
    "time_of_day_cn",
    "weather_cn",
    "is_ironic",
    "reasoning",
]
CHECK_ITEMS = [
    "fact_fidelity",
    "emotion_coverage",
    "metaphor_alignment",
    "visual_concreteness",
    "valence_arousal_reasonable",
    "time_weather_fidelity",
    "irony_or_mixed_emotion_handling",
    "prompt_readiness",
]


def parse_diaries(filepath):
    diaries = {}
    current_id = None
    current_title = ""
    current_text = []

    for line in filepath.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped[0].isdigit() and "." in stripped[:4]:
            if current_id:
                diaries[current_id] = {
                    "title": current_title,
                    "text": "\n".join(current_text).strip(),
                }
            num, title = stripped.split(".", 1)
            current_id = f"diary_{num.strip()}"
            current_title = title.strip()
            current_text = []
        else:
            current_text.append(line)

    if current_id:
        diaries[current_id] = {
            "title": current_title,
            "text": "\n".join(current_text).strip(),
        }
    return diaries


def schema_issues(emotion):
    issues = []
    for field in REQUIRED_FIELDS:
        if field not in emotion:
            issues.append(f"missing field: {field}")
    for field in ["secondary_emotions", "emotion_layers", "color_palette_cn", "scene_keywords_cn", "supporting_metaphors_cn"]:
        if field in emotion and not isinstance(emotion[field], list):
            issues.append(f"field should be list: {field}")
    for field in ["valence", "arousal", "intensity"]:
        if field in emotion and not isinstance(emotion[field], (int, float)):
            issues.append(f"field should be number: {field}")
    if isinstance(emotion.get("valence"), (int, float)) and not -1 <= emotion["valence"] <= 1:
        issues.append("valence out of range [-1, 1]")
    if isinstance(emotion.get("arousal"), (int, float)) and not 0 <= emotion["arousal"] <= 1:
        issues.append("arousal out of range [0, 1]")
    if isinstance(emotion.get("intensity"), (int, float)) and not 0 <= emotion["intensity"] <= 1:
        issues.append("intensity out of range [0, 1]")
    return issues


def build_entry(diary_id, diary, emotion, previous=None):
    previous = previous or {}
    checks = {name: "pending" for name in CHECK_ITEMS}
    checks.update(previous.get("checks", {}))
    issues = schema_issues(emotion)
    preserved_issues = [i for i in previous.get("issues", []) if not i.startswith("missing field:")]

    return {
        "approved": previous.get("approved", False),
        "failure_stage": previous.get("failure_stage", "manual_review_needed"),
        "reviewer_notes": previous.get("reviewer_notes", ""),
        "checks": checks,
        "issues": sorted(set(issues + preserved_issues)),
        "source": {
            "title": diary.get("title", ""),
            "text": diary.get("text", ""),
        },
        "extracted_summary": {
            "primary_emotion": emotion.get("primary_emotion"),
            "secondary_emotions": emotion.get("secondary_emotions"),
            "valence": emotion.get("valence"),
            "arousal": emotion.get("arousal"),
            "scene_keywords_cn": emotion.get("scene_keywords_cn"),
            "core_metaphor_cn": emotion.get("core_metaphor_cn"),
            "supporting_metaphors_cn": emotion.get("supporting_metaphors_cn"),
            "time_of_day_cn": emotion.get("time_of_day_cn"),
            "weather_cn": emotion.get("weather_cn"),
            "reasoning": emotion.get("reasoning"),
        },
        "decision_guide": {
            "approved_true_when": "JSON faithfully captures the diary emotion, does not invent factual scene details, and is ready for prompt construction.",
            "failure_stage_options": [
                "accepted",
                "prompt_instruction_unclear",
                "llm_model_capability_limit",
                "manual_review_needed",
            ],
            "if_failed": "Revise prompts/emotion_extraction_prompt.txt or switch extraction model, then rerun extract_emotions.py and audit again.",
        },
    }



def fmt_list(value):
    if not value:
        return "（空）"
    if isinstance(value, list):
        return "、".join(str(v) for v in value)
    return str(value)


def write_review_markdown(audit, output_path):
    lines = [
        "# Emotion JSON 人工审计表",
        "",
        "用途：检查 LLM 拆解出的 `emotions.json` 是否忠实原文。人工检查不是替代 LLM，而是判断前半段 pipeline 是否符合依据。",
        "",
        "判定建议：",
        "- 通过：在 `outputs/emotion_audit.json` 中设置 `approved: true`，`failure_stage: accepted`。",
        "- 不通过：保留 `approved: false`，并在 `failure_stage` 标注 `prompt_instruction_unclear` 或 `llm_model_capability_limit`。",
        "",
        "---",
        "",
    ]

    for diary_id, item in sorted(audit.items()):
        summary = item.get("extracted_summary", {})
        source = item.get("source", {})
        checks = item.get("checks", {})
        issues = item.get("issues", [])

        lines.extend([
            f"## {diary_id}：{source.get('title', '')}",
            "",
            "### 原文",
            "",
            source.get("text", "（空）"),
            "",
            "### LLM 拆解摘要",
            "",
            f"- 主情绪：{summary.get('primary_emotion')}",
            f"- 次要情绪：{fmt_list(summary.get('secondary_emotions'))}",
            f"- Valence / Arousal：{summary.get('valence')} / {summary.get('arousal')}",
            f"- 场景关键词：{fmt_list(summary.get('scene_keywords_cn'))}",
            f"- 核心隐喻：{summary.get('core_metaphor_cn')}",
            f"- 辅助隐喻：{fmt_list(summary.get('supporting_metaphors_cn'))}",
            f"- 时间 / 天气：{summary.get('time_of_day_cn')} / {summary.get('weather_cn')}",
            f"- Reasoning：{summary.get('reasoning')}",
            "",
            "### 自动检测到的问题",
            "",
        ])
        if issues:
            lines.extend(f"- {issue}" for issue in issues)
        else:
            lines.append("- 无格式层面的自动问题；仍需人工判断语义是否忠实。")

        lines.extend([
            "",
            "### 人工检查清单",
            "",
        ])
        check_labels = {
            "fact_fidelity": "事实忠实度：scene/time/weather 是否来自原文，没有编造空间、天气、人数或动作结果",
            "emotion_coverage": "情绪覆盖：是否抓住原文真正的核心情绪",
            "metaphor_alignment": "隐喻一致性：core/supporting metaphor 是否服务于原文，而不是另写故事",
            "visual_concreteness": "视觉可执行性：是否能转成清晰画面",
            "valence_arousal_reasonable": "效价/唤起度合理性",
            "time_weather_fidelity": "时间/天气忠实度",
            "irony_or_mixed_emotion_handling": "反讽/混合情绪处理",
            "prompt_readiness": "是否可以进入 main_prompt 构造",
        }
        for key in CHECK_ITEMS:
            status = checks.get(key, "pending")
            box = "x" if status == "pass" else " "
            lines.append(f"- [{box}] {check_labels[key]}（当前：{status}）")

        lines.extend([
            "",
            "### 人工结论",
            "",
            f"- 当前 approved：`{item.get('approved')}`",
            f"- 当前 failure_stage：`{item.get('failure_stage')}`",
            f"- reviewer_notes：{item.get('reviewer_notes') or '（待填写）'}",
            "",
            "建议填写：通过 / 不通过；如果不通过，问题属于提示词约束不清、LLM 能力不足，还是仍需人工判断。",
            "",
            "---",
            "",
        ])

    output_path.write_text("\n".join(lines), encoding="utf-8")

def main():
    emotions_path = BASE_DIR / "outputs" / "emotions.json"
    diaries_path = BASE_DIR / "data" / "diaries.txt"
    audit_path = BASE_DIR / "outputs" / "emotion_audit.json"
    review_path = BASE_DIR / "outputs" / "emotion_audit_review.md"

    if not emotions_path.exists():
        raise SystemExit("Missing outputs/emotions.json. Run scripts/extract_emotions.py first.")

    diaries = parse_diaries(diaries_path)
    emotions = json.loads(emotions_path.read_text(encoding="utf-8"))
    previous = {}
    if audit_path.exists():
        previous = json.loads(audit_path.read_text(encoding="utf-8"))

    audit = {}
    for diary_id, emotion in sorted(emotions.items()):
        audit[diary_id] = build_entry(
            diary_id,
            diaries.get(diary_id, {}),
            emotion,
            previous.get(diary_id),
        )

    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    write_review_markdown(audit, review_path)
    print(f"Wrote {audit_path}")
    print(f"Wrote {review_path}")
    print("Review the Markdown first. Set approved=true and failure_stage='accepted' in emotion_audit.json only after all checks pass.")


if __name__ == "__main__":
    main()
