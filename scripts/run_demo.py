#!/usr/bin/env python3
"""
run_demo.py — 端到端单条日记演示脚本

流程：
  日记文本 → DeepSeek（情绪提取）→ Prompt 构造 × 3 → DALL-E 3（图像生成）→ 保存图像

依赖：
  pip install openai deep-translator python-dotenv httpx requests

环境变量（写入项目根目录 .env）：
  DEEPSEEK_API_KEY=sk-xxx     # https://platform.deepseek.com
  OPENAI_API_KEY=sk-xxx       # https://platform.openai.com（DALL-E 3 需要 Tier 1+）

用法：
  python scripts/run_demo.py                   # 使用内置示例日记
  python scripts/run_demo.py --diary my.txt    # 使用自定义日记文件（UTF-8 中文）
"""

import argparse
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

import httpx
from dotenv import load_dotenv
from openai import OpenAI

# ── 路径 ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))

load_dotenv(dotenv_path=BASE_DIR / ".env")

# ── 示例日记（虚构，仅用于演示输入格式）────────────────────────────────────────
SAMPLE_DIARY = """今天复习到凌晨两点，脑子却像一团乱麻，什么都记不进去。\
翻来覆去地看同一页笔记，感觉字认识我，我不认识字。\
窗外楼道里有人大声打电话，笑声一阵一阵地传进来，我突然说不清楚自己是烦他吵，\
还是烦自己连安静地坐着都做不到。\
把书合上又打开，打开又合上，最后干脆趴在桌上，盯着台灯看了很久。\
努力了这么久，怎么还是觉得自己什么都抓不住？"""

# ── API 客户端 ────────────────────────────────────────────────────────────────
_proxy = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")

_ds_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    http_client=httpx.Client(proxy=_proxy) if _proxy else None,
)

_oai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    http_client=httpx.Client(proxy=_proxy) if _proxy else None,
)

# ── Step 1：情绪提取（DeepSeek）───────────────────────────────────────────────

def extract_emotion(diary_text: str) -> dict:
    prompt_path = BASE_DIR / "prompts" / "emotion_extraction_prompt.txt"
    template = prompt_path.read_text(encoding="utf-8")
    user_msg = template.replace("[USER_DIARY_CONTENT_HERE]", diary_text)

    response = _ds_client.chat.completions.create(
        model="deepseek-chat",
        max_tokens=1800,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": user_msg}],
    )
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```json?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


# ── Step 2：Prompt 构造 ───────────────────────────────────────────────────────
# 复用 build_prompts.py 中已有的构造函数，避免重复逻辑

from build_prompts import (  # noqa: E402
    GPT4O_PIXEL_STYLE,
    build_concise_prompt_cn,
    build_partial_prompt_en,
    clean_translated_en,
    preprocess_baseline,
    preprocess_full,
    translate_to_english,
)


def build_three_prompts(diary_text: str, emotion: dict) -> dict:
    """返回 {baseline, partial, full} 三条英文 prompt。"""
    # Partial：纯情绪标签 + valence/arousal，无需翻译
    partial = build_partial_prompt_en(emotion)

    # Full：中文结构化 prompt → Google Translate
    full_cn = build_concise_prompt_cn(emotion)
    full = clean_translated_en(translate_to_english(preprocess_full(full_cn)))

    # Baseline：原始日记 → Google Translate + 画风约束
    raw_en = clean_translated_en(translate_to_english(preprocess_baseline(diary_text)))
    baseline = f"{raw_en} {GPT4O_PIXEL_STYLE}."

    return {"baseline": baseline, "partial": partial, "full": full}


# ── Step 3：图像生成（DALL-E 3）────────────────────────────────────────────────

def generate_image(prompt: str, out_path: Path) -> str:
    """调用 DALL-E 3 生成图像，下载并保存到 out_path，返回本地路径字符串。"""
    response = _oai_client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality="standard",
        n=1,
    )
    url = response.data[0].url
    out_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, out_path)
    return str(out_path)


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="中文日记 → 像素艺术图像（三条件）")
    parser.add_argument(
        "--diary", type=str, default=None,
        help="日记文本文件路径（UTF-8），不传则使用内置示例"
    )
    parser.add_argument(
        "--out-dir", type=str, default="outputs/demo",
        help="图像输出目录（默认 outputs/demo）"
    )
    args = parser.parse_args()

    # 读取日记
    if args.diary:
        diary_text = Path(args.diary).read_text(encoding="utf-8").strip()
        print(f"已读取日记：{args.diary}")
    else:
        diary_text = SAMPLE_DIARY
        print("使用内置示例日记（diary_5）")

    out_dir = BASE_DIR / args.out_dir

    # Step 1
    print("\n[1/3] 情绪提取（DeepSeek）...")
    emotion = extract_emotion(diary_text)
    print(f"  主情绪: {emotion['primary_emotion']}")
    print(f"  核心隐喻: {emotion['core_metaphor_cn']}")
    print(f"  valence={emotion['valence']}  arousal={emotion['arousal']}")

    # Step 2
    print("\n[2/3] 构造三条件 Prompt...")
    prompts = build_three_prompts(diary_text, emotion)
    for cond, p in prompts.items():
        preview = p[:80] + "..." if len(p) > 80 else p
        print(f"  [{cond}] {preview}")

    # Step 3
    print("\n[3/3] 图像生成（DALL-E 3）...")
    results = {}
    for cond in ("baseline", "partial", "full"):
        print(f"  生成 {cond}...", end="", flush=True)
        path = out_dir / f"{cond}.png"
        saved = generate_image(prompts[cond], path)
        results[cond] = saved
        print(f" 已保存 → {saved}")

    # 摘要
    print("\n完成！生成图像：")
    for cond, path in results.items():
        print(f"  {cond}: {path}")

    # 保存本次运行的 prompt 记录
    log_path = out_dir / "prompts_used.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(
            {"emotion_summary": {
                "primary_emotion": emotion["primary_emotion"],
                "core_metaphor_cn": emotion["core_metaphor_cn"],
                "valence": emotion["valence"],
                "arousal": emotion["arousal"],
            }, "prompts": prompts},
            f, indent=2, ensure_ascii=False,
        )
    print(f"  prompts 记录: {log_path}")


if __name__ == "__main__":
    main()
