import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv
from openai import OpenAI

BASE_DIR = Path(__file__).parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

_ds_client = None


def _get_ds_client():
    global _ds_client
    if _ds_client is None:
        _proxy = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
        _ds_client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
            http_client=httpx.Client(proxy=_proxy),
        )
    return _ds_client

# GPT-4o 画风约束（所有三个条件共用，替代原 Civitai SD 的 STYLE_PREFIX/SUFFIX）
# 直接用英文构造，不依赖翻译，保证三个条件的风格干预量完全一致
GPT4O_PIXEL_STYLE = (
    "16-bit hard edge pixel art, limited color palette, clear pixel grid, "
    "no anti-aliasing, no readable text, no watermark, simple composition, centered subject"
)

TEXT_RISK_REPLACEMENTS = {
    "Boss直聘": "无文字的求职应用图标",
    "boss直聘": "无文字的求职应用图标",
    "厦门": "异乡街头",
    "广州": "回家路上",
    "惯性": "来回摇晃的动作",
    "哈哈哈哈": "开怀大笑的表情",
    "已读不回": "空白的已读消息气泡",
    "不匹配": "空的连接符号",
    "精打细算": "空的日程格子",
    "简历投递": "几张没有文字的空白纸",
    "简历": "没有文字的空白纸",
    "岗位描述": "没有文字的工作卡片",
    "A应用": "抽象的应用图标",
    "前后端": "两个分岔的抽象模块",
    "LeetCode 屏幕": "屏幕上的抽象谜题方块",
    "LeetCode屏幕": "屏幕上的抽象谜题方块",
    "leetcode": "抽象谜题方块",
    "Claude 聊天": "没有文字的发光聊天窗口",
    "Claude聊天": "没有文字的发光聊天窗口",
    "Claude": "没有文字的发光聊天窗口",
    "微信": "没有文字的手机消息界面",
    "WeChat": "没有文字的手机消息界面",
    "聊天框": "空白聊天气泡",
    "聊天记录": "空白聊天气泡",
    "终端": "带绿色成功光的电脑屏幕",
    "200 OK": "绿色成功勾号",
    "status 500": "红色错误符号",
    "Dify API": "抽象的 API 连接节点",
    "Dify": "抽象的 API 连接节点",
    "GitHub Issue": "没有文字的问题卡片",
    "代码报错": "红色错误符号",
    "代码": "抽象代码线条",
    "论文格式": "凌乱的空白页面",
    "朋友圈": "无文字的照片网格",
    "照片": "无文字的照片卡片",
    "落地窗": "明亮的落地窗",
}

NO_TEXT_CONSTRAINT_CN = (
    "画面中不要出现任何可读文字、字母、字幕、标志、水印或随机符号；"
    "屏幕和纸张只能显示抽象图形或空白形状。"
)

# 已知会产生字面直译问题的中文成语/比喻
PROBLEMATIC_IDIOMS = [
    "无头苍蝇", "狗皮膏药", "热锅上的蚂蚁", "墙头草",
    "如鱼得水", "鸭子听雷", "丈二和尚", "一头雾水",
]


def visual_safe_element(element):
    text = str(element).strip()
    for key, replacement in TEXT_RISK_REPLACEMENTS.items():
        if key.lower() in text.lower():
            return replacement
    return text


def clean_optional(value):
    value = (value or "").strip()
    if value in {"未明确", "无", "无明确", "不明确", "未提及", "无提及", "N/A", "n/a", "unknown", "none", "Unclarified"}:
        return ""
    return value


def mood_from_affect(valence, arousal):
    if valence > 0.5 and arousal > 0.5:
        return "明亮、轻快、有释放感"
    if valence > 0.5 and arousal <= 0.5:
        return "安静、温暖、松弛"
    if valence > -0.3 and valence <= 0.5 and arousal > 0.6:
        return "荒诞、起伏、出其不意"
    if valence > -0.3 and valence <= 0.5:
        return "内省、柔和、带一点沉默"
    if valence <= -0.3 and arousal > 0.5:
        return "紧张、压抑、不安"
    return "低沉、缓慢、忧郁"


def valence_level(v):
    return "high" if v > 0.3 else ("low" if v < -0.3 else "mid")


def arousal_level(a):
    return "high" if a > 0.6 else ("low" if a < 0.3 else "mid")


# ─────────────────────────────────────────────────────────────────────────────
# 三个条件的 prompt 构造函数
# ─────────────────────────────────────────────────────────────────────────────

def build_concise_prompt_cn(emotion_json):
    """Full 条件：完整结构化中文 prompt，包含隐喻 + 色彩 + 场景 + 情绪 + 构图。

    自然语言段落格式，适合 GPT-4o 理解（不是 SD tag 格式）。
    翻译为英文后作为 Full 条件发送。
    """
    e = emotion_json
    mood = mood_from_affect(e["valence"], e["arousal"])
    scenes = [visual_safe_element(x) for x in e.get("scene_keywords_cn", [])[:3]]
    colors = e.get("color_palette_cn", [])[:2]
    supporting = e.get("supporting_metaphors_cn", [])
    inner_metaphor = supporting[0] if supporting else ""
    emotion_terms = "、".join(
        [e.get("primary_emotion", "")] + e.get("secondary_emotions", [])[:2]
    )
    env_part = "、".join(
        x for x in [
            clean_optional(e.get("time_of_day_cn", "")),
            clean_optional(e.get("weather_cn", ""))
        ] if x
    )
    scene_sentence = "、".join(scenes) if scenes else "少量象征性物件"
    color_sentence = "、".join(colors) if colors else "克制的情绪色调"

    parts = [
        "一幅16位复古像素艺术场景，构图简洁，主体明确。",
        f"画面中心呈现：{e['core_metaphor_cn']}。",
        f"场景中只保留少量关键物件：{scene_sentence}。",
    ]
    if inner_metaphor:
        parts.append(f"画面通过这个视觉暗示表达内在感受：{inner_metaphor}。")
    if env_part:
        parts.append(f"环境线索：{env_part}。")
    parts.extend([
        f"整体氛围是{mood}，对应情绪包括：{emotion_terms}。",
        f"色彩使用{color_sentence}，避免过多装饰。",
        NO_TEXT_CONSTRAINT_CN,
    ])
    return "\n".join(parts)


def build_partial_prompt_en(emotion_json):
    """Partial 条件：仅情绪标签 + valence/arousal，无视觉隐喻和色彩指定。

    直接构造英文，不需要 Google Translate（字段值已为英文）。
    验证问题：LLM 情绪语义提取本身是否有效？
    """
    e = emotion_json
    v = valence_level(e["valence"])
    a = arousal_level(e["arousal"])

    primary = e.get("primary_emotion", "")
    secondary = [s for s in e.get("secondary_emotions", [])[:2] if s]
    all_emotions = [em for em in [primary] + secondary if em]
    emotions_str = ", ".join(all_emotions)

    return f"{GPT4O_PIXEL_STYLE}, {emotions_str}, {v} valence, {a} arousal"


def build_greedy_prompt_cn(emotion_json):
    """贪心版：包含所有 JSON 层次，用于信息密度对比（历史保留，非主实验）。"""
    e = emotion_json
    all_scenes = "、".join(e["scene_keywords_cn"])
    all_colors = "、".join(e["color_palette_cn"])
    supporting = "、".join(e.get("supporting_metaphors_cn", []))
    layers_desc = "，".join([l["description"] for l in e.get("emotion_layers", [])])
    env_part = "、".join(
        x for x in [e.get("time_of_day_cn", ""), e.get("weather_cn", "")] if x
    )
    return (
        f"像素艺术，多层情绪可视化：\n"
        f"核心隐喻：{e['core_metaphor_cn']}\n"
        f"辅助元素：{supporting}\n"
        f"画面元素：{all_scenes}\n"
        f"情绪层次：{layers_desc}\n"
        f"色调：{all_colors}\n"
        + (f"时间天气：{env_part}\n" if env_part else "")
        + "16位复古游戏风格"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 翻译
# ─────────────────────────────────────────────────────────────────────────────

def translate_to_english(chinese_text: str) -> str:
    """机械翻译：Google Translate（非 LLM），保证不添加情绪色彩。"""
    from deep_translator import GoogleTranslator

    def translate_once(text):
        last_error = None
        for attempt in range(1, 4):
            try:
                return GoogleTranslator(source="zh-CN", target="en").translate(text)
            except Exception as exc:
                last_error = exc
                if attempt < 3:
                    time.sleep(1.5 * attempt)
        raise last_error

    normalized = chinese_text.replace("\n", "，").strip("，")
    MAX_CHARS = 4500
    if len(normalized) <= MAX_CHARS:
        result = translate_once(normalized)
    else:
        parts = [p.strip() for p in normalized.split("，") if p.strip()]
        chunks, current = [], ""
        for part in parts:
            if len(current) + len(part) + 1 > MAX_CHARS:
                chunks.append(current.strip("，"))
                current = part
            else:
                current += "，" + part
        if current:
            chunks.append(current.strip("，"))
        result = ", ".join(translate_once(c) for c in chunks)

    # 修复 Google Translate 中文句子末尾导致的 "., " 格式问题
    result = re.sub(r"\.\s*,\s*", ". ", result)
    result = re.sub(r"\s+", " ", result).strip()
    return result


BASELINE_PRETRANSLATE = {
    "毕设": "毕业论文",
    "毕业设计": "毕业论文",
}

# 可能被机械翻译误解为暴力的中文动作词（避免使用单字"打"/"撞"以防误匹配）
CONTEXT_SENSITIVE_WORDS = [
    "撞向", "撞入", "撞上",
    "推倒", "推搡",
    "砸向", "揍", "殴打",
]


def rewrite_sensitive_cn(text: str) -> str:
    """当中文 prompt 含可能被机械翻译误解为暴力的词汇时，
    调用 DeepSeek 根据上下文改写为无歧义表达，再交给 Google Translate。
    如果不含敏感词或 API 失败，直接返回原文。
    """
    found = [w for w in CONTEXT_SENSITIVE_WORDS if w in text]
    if not found:
        return text

    system_msg = (
        "你是一位精准的中文改写助手，服务于 AI 绘图 prompt 生成流程。"
        "任务：将以下中文描述中可能被机械翻译成英文时产生歧义或暴力联想的词汇，"
        "在完全保留原意和情感不变的前提下，改写为语义更准确、无歧义的中文表达。"
        "只修改有歧义的词汇或短语，其余内容一字不改。"
        "直接返回改写后的完整文本，不要添加任何解释。"
    )
    user_msg = f"发现可能歧义词：{', '.join(found)}\n\n原文：\n{text}"

    try:
        resp = _get_ds_client().chat.completions.create(
            model="deepseek-chat",
            max_tokens=600,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        print(f"  ⚠ rewrite_sensitive_cn 失败，使用原文: {exc}")
        return text


def preprocess_baseline(text: str) -> str:
    for k, v in BASELINE_PRETRANSLATE.items():
        text = text.replace(k, v)
    return text


def preprocess_full(text: str) -> str:
    return rewrite_sensitive_cn(text)


def clean_translated_en(text: str) -> str:
    """清理翻译后的英文：移除多余逗号/句号组合，标准化空格。"""
    text = re.sub(r"\.\s*,\s*", ". ", text)
    text = re.sub(r",\s*\.", ".", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# 审计门禁
# ─────────────────────────────────────────────────────────────────────────────

def validate_emotion_audit(emotions, skip_audit=False):
    if skip_audit:
        print("⚠ 跳过 emotion_audit.json 门禁（仅调试用）")
        return

    audit_path = BASE_DIR / "outputs" / "emotion_audit.json"
    if not audit_path.exists():
        print("✗ 未找到 outputs/emotion_audit.json，请先运行 audit_emotions.py")
        sys.exit(1)

    with open(audit_path, "r", encoding="utf-8") as f:
        audit = json.load(f)

    blocked = []
    for diary_id in sorted(emotions):
        item = audit.get(diary_id)
        if not item:
            blocked.append(f"{diary_id}: 缺少审计记录")
            continue
        if item.get("approved") is not True or item.get("failure_stage") != "accepted":
            stage = item.get("failure_stage", "unknown")
            notes = item.get("reviewer_notes", "")
            blocked.append(f"{diary_id}: approved={item.get('approved')} stage={stage} notes={notes}")

    if blocked:
        print("✗ 情绪 JSON 尚未全部通过人工审计，停止生成 prompt：")
        for line in blocked:
            print(f"  - {line}")
        sys.exit(1)

    print("✓ emotion_audit.json 全部通过，开始构造 prompts")


# ─────────────────────────────────────────────────────────────────────────────
# 问题检测（辅助人工复审）
# ─────────────────────────────────────────────────────────────────────────────

def detect_issues(emotion_json):
    """检测 prompt 中可能影响生成质量的问题，辅助人工复审。"""
    issues = []
    e = emotion_json

    # 成语/比喻直译问题
    text_to_check = (
        e.get("core_metaphor_cn", "")
        + "".join(e.get("supporting_metaphors_cn", []))
        + "".join(e.get("scene_keywords_cn", []))
    )
    for idiom in PROBLEMATIC_IDIOMS:
        if idiom in text_to_check:
            issues.append(f"⚠ 包含成语「{idiom}」—— 机械翻译可能失义，建议手动改写隐喻")

    # 时间/天气字段（可能是编造）
    time_val = clean_optional(e.get("time_of_day_cn", ""))
    weather_val = clean_optional(e.get("weather_cn", ""))
    if time_val or weather_val:
        issues.append(
            f"ℹ 时间={time_val or '无'} / 天气={weather_val or '无'}，"
            f"请确认来自原文而非 LLM 推断"
        )

    # 反讽日记
    if e.get("is_ironic"):
        issues.append("⚠ 反讽日记（is_ironic=True）—— 图像模型难以呈现反讽，建议在论文中单独讨论")

    # valence/arousal 超出合理范围
    v, a = e.get("valence", 0), e.get("arousal", 0)
    if not (-1 <= v <= 1):
        issues.append(f"✗ valence={v} 超出 [-1, 1] 范围")
    if not (0 <= a <= 1):
        issues.append(f"✗ arousal={a} 超出 [0, 1] 范围")

    return issues


# ─────────────────────────────────────────────────────────────────────────────
# 实时预览打印
# ─────────────────────────────────────────────────────────────────────────────

def print_diary_preview(diary_id, emotion_json, partial_en, full_en, baseline_en=""):
    """处理完每条日记后立即打印三个条件的 prompt，辅助发现问题。"""
    e = emotion_json
    v = e["valence"]
    a = e["arousal"]
    vl = valence_level(v)
    al = arousal_level(a)
    ironic_tag = "  [反讽]" if e.get("is_ironic") else ""

    sep = "─" * 68
    print(f"\n{sep}")
    print(
        f"【{diary_id}】 {e['primary_emotion']}{ironic_tag}"
        f"  valence={v} ({vl})  arousal={a} ({al})"
    )
    print(f"  核心隐喻: {e['core_metaphor_cn']}")
    secondary = e.get("secondary_emotions", [])
    if secondary:
        print(f"  次要情绪: {', '.join(secondary)}")

    issues = detect_issues(e)
    for iss in issues:
        print(f"  {iss}")

    print(f"\n  ▸ [Partial]")
    print(f"    {partial_en}")

    print(f"\n  ▸ [Full]")
    # 按 80 字符换行显示，方便阅读
    words = full_en.replace(". ", ".\n    ").split("\n")
    for line in words:
        print(f"    {line.strip()}")

    if baseline_en:
        preview = baseline_en[:120] + ("..." if len(baseline_en) > 120 else "")
        print(f"\n  ▸ [Baseline] (前 120 字)")
        print(f"    {preview}")

    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# 解析日记
# ─────────────────────────────────────────────────────────────────────────────

def parse_diaries(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    diaries = {}
    current_id, current_text = None, []
    for line in content.strip().split("\n"):
        stripped = line.strip()
        m = re.match(r"^(\d+)\.", stripped)
        if m:
            if current_id and current_text:
                diaries[current_id] = "\n".join(current_text).strip()
            current_id, current_text = f"diary_{m.group(1)}", []
        elif stripped:
            current_text.append(line)
    if current_id and current_text:
        diaries[current_id] = "\n".join(current_text).strip()
    return diaries


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-audit", action="store_true",
        help="调试用：跳过 emotion_audit 门禁"
    )
    parser.add_argument(
        "--no-greedy", action="store_true",
        help="跳过贪心版 prompt 生成（节省时间）"
    )
    args = parser.parse_args()

    emotions_path = BASE_DIR / "outputs" / "emotions.json"
    if not emotions_path.exists():
        print("✗ 未找到 outputs/emotions.json，请先运行 extract_emotions.py")
        return

    with open(emotions_path, "r", encoding="utf-8") as f:
        emotions = json.load(f)

    validate_emotion_audit(emotions, skip_audit=args.skip_audit)

    # 解析原始日记（用于 Baseline）
    diaries = {}
    diaries_path = BASE_DIR / "data" / "diaries.txt"
    if diaries_path.exists():
        diaries = parse_diaries(diaries_path)
    else:
        print("⚠ 未找到 data/diaries.txt，Baseline 条件将跳过")

    # ── 逐条处理，实时打印结果 ──
    print("\n构造三条件 prompts（Baseline / Partial / Full）...\n")

    main_prompts = {}
    partial_prompts = {}
    english_prompts = {}
    baseline_english = {}

    for diary_id in sorted(emotions.keys()):
        emotion = emotions[diary_id]
        print(f"处理 {diary_id}...", end="", flush=True)

        # Partial：直接构造英文，无需翻译
        partial_en = build_partial_prompt_en(emotion)

        # Full：中文 → Google Translate
        concise_cn = build_concise_prompt_cn(emotion)
        full_en = clean_translated_en(translate_to_english(preprocess_full(concise_cn)))

        # Baseline：原始日记 → Google Translate → 加画风约束
        baseline_en = ""
        if diary_id in diaries:
            raw_en = clean_translated_en(translate_to_english(preprocess_baseline(diaries[diary_id])))
            baseline_en = f"{raw_en} {GPT4O_PIXEL_STYLE}."

        # 存储
        emotion_summary = {
            "primary": emotion["primary_emotion"],
            "valence": emotion["valence"],
            "arousal": emotion["arousal"],
            "core_metaphor": emotion["core_metaphor_cn"],
            "is_ironic": emotion.get("is_ironic", False),
        }
        main_prompts[diary_id] = {
            "emotion_summary": emotion_summary,
            "concise_prompt": concise_cn,
        }
        partial_prompts[diary_id] = {
            "emotion_summary": emotion_summary,
            "partial_prompt": partial_en,
        }
        english_prompts[diary_id] = {
            "emotion_summary": emotion_summary,
            "english_prompt": full_en,
        }
        if baseline_en:
            baseline_english[diary_id] = {"english_prompt": baseline_en}

        print(" ✓")

        # 实时预览
        print_diary_preview(diary_id, emotion, partial_en, full_en, baseline_en)

    # ── 贪心版（历史保留，非主实验） ──
    if not args.no_greedy:
        print("\n构造贪心版 prompts（历史保留，非主实验）...")
        greedy_prompts = {}
        greedy_english_prompts = {}
        for diary_id, emotion in sorted(emotions.items()):
            print(f"  {diary_id}...", end="", flush=True)
            greedy_cn = build_greedy_prompt_cn(emotion)
            greedy_en = clean_translated_en(translate_to_english(greedy_cn))
            greedy_prompts[diary_id] = {"greedy_prompt": greedy_cn}
            greedy_english_prompts[diary_id] = {"greedy_english_prompt": greedy_en}
            print(" ✓")

        with open(BASE_DIR / "outputs" / "greedy_prompts.json", "w", encoding="utf-8") as f:
            json.dump(greedy_prompts, f, indent=2, ensure_ascii=False)
        with open(BASE_DIR / "outputs" / "greedy_english_prompts.json", "w", encoding="utf-8") as f:
            json.dump(greedy_english_prompts, f, indent=2, ensure_ascii=False)

    # ── 保存所有文件 ──
    outputs = BASE_DIR / "outputs"
    with open(outputs / "main_prompts.json", "w", encoding="utf-8") as f:
        json.dump(main_prompts, f, indent=2, ensure_ascii=False)
    with open(outputs / "partial_prompts.json", "w", encoding="utf-8") as f:
        json.dump(partial_prompts, f, indent=2, ensure_ascii=False)
    with open(outputs / "english_prompts.json", "w", encoding="utf-8") as f:
        json.dump(english_prompts, f, indent=2, ensure_ascii=False)
    if baseline_english:
        with open(outputs / "baseline_english.json", "w", encoding="utf-8") as f:
            json.dump(baseline_english, f, indent=2, ensure_ascii=False)

    print("\n✓ 所有文件已保存：")
    print("  outputs/partial_prompts.json    ← Partial 条件（新增）")
    print("  outputs/english_prompts.json    ← Full 条件")
    print("  outputs/baseline_english.json   ← Baseline 条件")
    print("  outputs/main_prompts.json       ← 中文版（备用）")
    if not args.no_greedy:
        print("  outputs/greedy_prompts.json     ← 贪心版（历史保留）")
        print("  outputs/greedy_english_prompts.json")


if __name__ == "__main__":
    main()
