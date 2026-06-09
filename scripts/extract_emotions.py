import json
import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import httpx

BASE_DIR = Path(__file__).parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

# ALL_PROXY 可能使用 socks:// 格式，httpx 不支持直接解析。
# 显式使用 HTTPS_PROXY（http:// 格式）并忽略 ALL_PROXY，避免初始化报错。
_proxy = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    http_client=httpx.Client(proxy=_proxy),
)


def parse_diaries(filepath):
    """解析 diaries.txt，按数字编号行（如 "1. 标题"）分隔"""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    diaries = {}
    current_id = None
    current_text = []

    for line in content.strip().split("\n"):
        stripped = line.strip()
        # 匹配 "N. " 开头的标题行作为分隔符
        import re
        if re.match(r"^\d+\.", stripped):
            if current_id and current_text:
                diaries[current_id] = "\n".join(current_text).strip()
            num = re.match(r"^(\d+)\.", stripped).group(1)
            current_id = f"diary_{num}"
            current_text = []
        elif stripped:
            current_text.append(line)

    if current_id and current_text:
        diaries[current_id] = "\n".join(current_text).strip()

    return diaries


def clean_json_response(text):
    """清理 LLM 返回的 markdown 围栏"""
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()
    return text


def chat_json(messages, max_tokens=1800):
    """Request a JSON object. Fall back if the provider rejects response_format."""
    kwargs = {
        "model": "deepseek-chat",
        "max_tokens": max_tokens,
        "messages": messages,
    }
    try:
        response = client.chat.completions.create(
            **kwargs,
            response_format={"type": "json_object"},
        )
    except Exception as exc:
        if "response_format" not in str(exc):
            raise
        response = client.chat.completions.create(**kwargs)
    return clean_json_response(response.choices[0].message.content)


def repair_json_response(invalid_json, error_message):
    """Ask the model to repair syntax only; do not change content."""
    repair_prompt = f"""
下面是一段解析失败的 JSON。请只修复 JSON 语法，保持字段和含义不变，不要添加解释文本。
特别注意：字符串内部的英文双引号必须转义，或改成中文引号「」。

解析错误：{error_message}

待修复 JSON：
{invalid_json}
""".strip()
    return chat_json([{"role": "user", "content": repair_prompt}], max_tokens=1800)


def main():
    diaries = parse_diaries(BASE_DIR / "data" / "diaries.txt")
    print(f"解析到 {len(diaries)} 条日记：{list(diaries.keys())}")

    with open(BASE_DIR / "prompts" / "emotion_extraction_prompt.txt", "r", encoding="utf-8") as f:
        prompt_template = f.read()

    emotions = {}
    for diary_id, diary_text in sorted(diaries.items()):
        print(f"\n处理 {diary_id}...")
        user_prompt = prompt_template.replace("[USER_DIARY_CONTENT_HERE]", diary_text)

        try:
            response_text = chat_json([{"role": "user", "content": user_prompt}])
            try:
                emotion_json = json.loads(response_text)
            except json.JSONDecodeError as e:
                print(f"  ⚠ JSON 初次解析失败，尝试语法修复: {e}")
                repaired_text = repair_json_response(response_text, str(e))
                emotion_json = json.loads(repaired_text)
                response_text = repaired_text
            emotions[diary_id] = emotion_json

            print(f"  ✓ 主情绪: {emotion_json['primary_emotion']}")
            print(f"  ✓ 核心隐喻: {emotion_json['core_metaphor_cn']}")
            print(f"  ✓ valence={emotion_json['valence']}, arousal={emotion_json['arousal']}")
            if emotion_json.get("is_ironic"):
                print(f"  ⚠ 反讽识别: True")

        except json.JSONDecodeError as e:
            print(f"  ✗ JSON 解析失败: {e}")
            raw = response_text if "response_text" in dir() else "(无响应)"
            print(f"  原始响应（前 300 字）: {raw[:300]}")
        except Exception as e:
            print(f"  ✗ 错误: {e}")

    output_path = BASE_DIR / "outputs" / "emotions.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(emotions, f, indent=2, ensure_ascii=False)

    print(f"\n✓ 已保存 {len(emotions)} 条情绪到 outputs/emotions.json")


if __name__ == "__main__":
    main()
