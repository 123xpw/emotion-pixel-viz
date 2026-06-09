import csv
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

BASE_DIR = Path(__file__).parent.parent
CONDITIONS = ["condition_baseline", "condition_partial", "condition_full"]

# 英文情绪参考文本，用于 CLIP 图文对齐评分
REFERENCES = {
    "diary_1": "anxiety and confusion at career crossroads, scattered job applications, self-doubt",
    "diary_2": "loneliness and self-blame, studying alone on a sunny day, feeling burdensome to others",
    "diary_3": "pure joy and freedom, laughing child bouncing on subway, carefree happiness",
    "diary_4": "warm nostalgia and longing, grandparents scent, deep emotional dependence and love",
    "diary_5": "anger and frustration after being belittled in interview, humiliation, inability to speak up",
    "diary_6": "calm and relief, walking alone on track at night, cool breeze, peaceful acceptance",
    "diary_7": "bittersweet laughter in chaos, late night lab stress, absurd cat cry releasing tension",
    "diary_8": "ironic anxiety masked as freedom, helplessness disguised as possibility, inner despair",
    "diary_9": "fear of being left behind, peer pressure, watching friend succeed while feeling stuck",
    "diary_10": "triumph and control, solving a bug after three days, pure joy of mastery",
}


def compute_clip_score(model, processor, image_path, text):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=[text], images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        img_feat = F.normalize(
            model.visual_projection(model.vision_model(pixel_values=inputs["pixel_values"]).pooler_output),
            dim=-1,
        )
        txt_feat = F.normalize(
            model.text_projection(
                model.text_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).pooler_output
            ),
            dim=-1,
        )
        return (img_feat * txt_feat).sum().item()


def main():
    print("加载 CLIP 模型...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()

    rows = []
    condition_totals = {c: [] for c in CONDITIONS}

    for diary_id, ref_text in sorted(REFERENCES.items()):
        print(f"\n{diary_id}：{ref_text[:40]}...")
        for condition in CONDITIONS:
            img_path = BASE_DIR / "outputs" / condition / f"{diary_id}.png"
            if not img_path.exists():
                print(f"  {condition}: 文件不存在，跳过")
                continue
            score = compute_clip_score(model, processor, img_path, ref_text)
            condition_totals[condition].append(score)
            rows.append({"diary": diary_id, "condition": condition, "clip_score": round(score, 6)})
            label = condition.replace("condition_", "")
            print(f"  {label:10s}: {score:.4f}")

    # 保存逐条结果
    out_path = BASE_DIR / "outputs" / "clip_scores.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["diary", "condition", "clip_score"])
        writer.writeheader()
        writer.writerows(rows)

    # 打印各条件均值
    print("\n===== 各条件 CLIP 均值 =====")
    for condition in CONDITIONS:
        scores = condition_totals[condition]
        if scores:
            label = condition.replace("condition_", "")
            print(f"  {label:10s}: {sum(scores)/len(scores):.4f}  (n={len(scores)})")

    print(f"\n结果已保存到 outputs/clip_scores.csv")


if __name__ == "__main__":
    main()
