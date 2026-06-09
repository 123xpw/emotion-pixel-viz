"""
自动创建 Google Form 偏好测试问卷。
图片托管：Imgur（免费匿名上传）
运行：venv/bin/python scripts/create_google_form.py
"""

import os
import json
import base64
import requests as req

from google.oauth2 import service_account
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/forms.body"]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
UPLOAD_DIR = os.path.join(OUTPUTS_DIR, "upload_to_drive")
SA_FILE = os.path.join(BASE_DIR, "service_account.json")
TOKEN_FILE = os.path.join(BASE_DIR, ".github_token")

GITHUB_REPO = "123xpw/emotion-pixel-viz"
GITHUB_BRANCH = "main"
GITHUB_IMAGE_DIR = "survey_images"

# A/B/C → 条件映射（seed=42 生成）
ABC_MAP = {
    1:  {"A": "condition_partial",  "B": "condition_baseline", "C": "condition_full"},
    2:  {"A": "condition_full",     "B": "condition_partial",  "C": "condition_baseline"},
    3:  {"A": "condition_partial",  "B": "condition_full",     "C": "condition_baseline"},
    4:  {"A": "condition_partial",  "B": "condition_full",     "C": "condition_baseline"},
    5:  {"A": "condition_partial",  "B": "condition_baseline", "C": "condition_full"},
    6:  {"A": "condition_baseline", "B": "condition_partial",  "C": "condition_full"},
    7:  {"A": "condition_partial",  "B": "condition_full",     "C": "condition_baseline"},
    8:  {"A": "condition_partial",  "B": "condition_full",     "C": "condition_baseline"},
    9:  {"A": "condition_partial",  "B": "condition_full",     "C": "condition_baseline"},
    10: {"A": "condition_partial",  "B": "condition_baseline", "C": "condition_full"},
}

DIARIES = {
    1: (
        "题目 1：阴晴不定的 22 岁（焦虑迷茫）",
        """我的 22岁可以形容为阴晴不定的天气（焦虑迷茫）

这两天吧 依旧在 boss直聘投递简历 我发现自己投递依旧五花八门的 没有具体的方向 但它可能只要出现A应用我就会投上次还投递了一个前后端的(我就是乱来的)
她问我是否考虑前端 我问了岗位描述之后并目询问这个面试考察范围是否会很紧什么的 然后我又问了是否有 AI应用的方向 她后续没有回复我 我又陷入了沉思
就是外界可能一个变化 我开始在想另一条路的可能性 如果我努力的面试 真的进了 工作内容不喜欢呢 发展好像不是很好了，可是他是字节呀我其实最近又在怀疑是否不读研是一个错误? 因为陆续很多岗位都要求硕士 我就开始产生怀疑自己的选择是否是正确的"""
    ),
    2: (
        "题目 2：周六出了大太阳（孤独自责）",
        """周六出了大太阳（孤独自责）

这个时候应该好好的休息我选择了去教学楼刷leetcode研究dify项目感觉自己的未来没有什么盼头 因为自己现在明确了大致的方向 但是没有一个面试 而且就算面试,我也不知道要怎么面对。。 一早上和 Claude聊了很多 从是否真的直接这个AI产品的方向 到我的焦虑问题 最后到毕设导师选择 以及推荐话术。。
我自己很难受的是对那些聊过几次天的人无法正常的开口说话 感觉在某种程度上 我们不应该继续聊天了 他们是没有麻烦过我的
我总是麻烦别人 总觉得是令人讨厌的 但是令人讨厌又怎么样 我总是在小心翼翼 这样也太累了 很没有必要 你还是开心一点 也看开吧"""
    ),
    3: (
        "题目 3：地铁上的小女孩（纯粹快乐）",
        """地铁一个小女孩能够连续笑5分钟（纯粹快乐）

她就来来回回的去撞她爸爸因为地铁使她有惯性像反方向动 这会让她像在飞翔一样 她一直重复着这个过程 一直在哈哈哈哈哈哈很自由地笑着 好幸福啊她"""
    ),
    4: (
        "题目 4：我一生都会怀念的气味（怀念温暖）",
        """我一生都会怀念的气味（怀念温暖）

今天来厦门找好朋友 收到爷爷给我发的微信 他说要给我一些钱 还嘱咐我要买早上的票回广州.我一直强调不需要 爷爷你可以自己买点好吃的
我脑袋放空,却隐隐传来了爷爷奶奶身上的味道我想一辈子都会喜欢和怀念这股味道 我是多么的依靠他们 或许在后来的任何地方
我都可能很难再闻到这样的味道"""
    ),
    5: (
        "题目 5：无法反驳的下午（愤怒挫败）",
        """无法反驳的下午（愤怒挫败）

今天下午的面谈让我到现在心里都堵得慌。对方是个看起来挺资深的HR，但他全程几乎没怎么抬头看我，一直在敲键盘，甚至在我自我介绍到一半时打断我，用一种非常轻飘飘的语气说："现在的应届生怎么都喜欢在简历里写'熟悉AI和大模型'？跑个Demo也算熟悉吗？"那一瞬间，我感觉脸上一阵滚烫，不仅是尴尬，更多的是一种被粗暴否定的愤怒。我为了弄懂Dify和API调用，熬了好几个通宵，查了无数文档，在他眼里居然成了"装点门面的垃圾"。回学校的地铁上，我一直紧紧捏着手机，气得手都在抖。我气他的傲慢和不尊重，但更让我感到挫败的是，我当时居然一句话都没反驳，只是唯唯诺诺地笑了笑。为什么我连捍卫自己努力的勇气都没有？"""
    ),
    6: (
        "题目 6：操场的风（平静释然）",
        """操场的风（平静释然）

晚上十点，我关掉了Boss直聘，决定今晚不再投任何简历，也不再去想毕设和导师的事。我独自去操场走了五圈。今晚的风凉凉的，吹在身上很舒服，橡胶跑道上还有点白天暴晒后的余温。看着旁边几个在夜跑和低声聊天的人，我突然觉得，那些让我失眠的"字节面试"、"硕士学历"、"未来的出路"，在这一刻好像都离我很远。就算我暂时找不到完美的工作，明天的太阳也还是会照样升起，我现在的呼吸也依然是真实存在的。我没必要在今晚就把一生的难题都解答出来。走完最后一圈，心里竟然有种久违的轻松。就这样吧，一步一步来。"""
    ),
    7: (
        "题目 7：实验室的猫叫声（苦中作乐）",
        """实验室的猫叫声（复杂混合/苦中作乐）

今晚和室友在实验室加班到十一点半，我的论文格式调得一团糟，她的代码也一直在报莫名其妙的错。我们两个灰头土脸地坐在位子上，外卖送来的柠檬茶还漏了大半杯，桌上一片黏糊糊的。就在我烦躁得想砸键盘的时候，窗外突然传来一声特别凄惨又怪异的猫叫，听起来简直跟我们那个脾气暴躁的班主任训人时的尾音一模一样。我和室友对视了一眼，突然爆发出了一阵狂笑，笑到眼泪都出来了，甚至得捂着肚子蹲在地上。明明两个人都前途未卜、很迷茫，但那一刻的荒诞和快乐又是那么真实。生活已经这么苦了，能抓住一点乐子就先笑出来再说吧。"""
    ),
    8: (
        "题目 8：被羡慕的自由（反讽）",
        """被羡慕的自由（反讽隐晦/表面正面实际负面）

小姨今天在微信上给我转了一个"22岁必去的十个旅行目的地"的链接，还发语音羡慕地对我说："真羡慕你们年轻人呀，现在刚毕业，不用背房贷，不用带孩子，每天都有大把自由支配的时间，想去哪就去哪，生活真是充满了无限可能！"我看着屏幕，发了一个"谢谢小姨"加开心转圈的表情包。是啊，我多自由啊。我自由到可以每天花八个小时盯着毫无动静的聊天框，自由到可以在"未读"、"已读不回"和"不匹配"之间做选择，自由到连明天住哪里、吃什么都要精打细算。这种"充满无限可能"的自由，真是让人幸福得想哭。"""
    ),
    9: (
        "题目 9：站台上的行李（同辈压力）",
        """站台上的行李（第二种焦虑/同辈压力与掉队恐慌）

今天刷朋友圈，看到高中玩得最好的朋友发了她在上海新租的公寓照片。精巧的落地窗，高耸的写字楼，还有她和新同事在西餐厅的合影。曾经我们是一起喝15块钱奶茶、为数学周测发愁的人，而现在，她已经利落地迈入了"社会人"的轨道，过得体面又充实。看着那些照片，我的心突然像被针扎了一下，涌上一股极其强烈的恐慌。这种焦虑和找不找得到工作不一样，它是一种"被留下的恐惧"。我害怕大家都在往前跑，坐上了通往成熟的快车，而我还在站台上抓着行李手足无措。我们以后是不是会越来越没有共同话题？我是不是注定要被留在过去？"""
    ),
    10: (
        "题目 10：200 OK（掌控感与成就感）",
        """200 OK（第二种快乐/掌控感与成就感）

终于！那个折磨了我三天的Dify API连接报错（status 500）今晚终于被我解决了！下午我强迫自己静下心来，一行行对照GitHub上的Issue，尝试了六种不同的环境配置。当我最后一次按下运行键，看到终端里终于跳出绿色的"200 OK"，并且AI助手顺畅地吐出回答的那一瞬间，我直接在椅子上蹦了起来，甚至在空无一人的宿舍里握拳小声喊了一句"Yes！"。在这个失控、充满变数的世界里，至少有些东西，只要我付出了时间和努力，它就真的会给我对的回报。"""
    ),
}


def authenticate():
    creds = service_account.Credentials.from_service_account_file(
        SA_FILE, scopes=SCOPES
    )
    return creds


def upload_images_to_github(token):
    """上传 30 张图到 GitHub，返回 {filename: raw_url}"""
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }
    file_urls = {}
    for i in range(1, 11):
        for letter in ["A", "B", "C"]:
            filename = f"{i}{letter}.png"
            filepath = os.path.join(UPLOAD_DIR, filename)
            if not os.path.exists(filepath):
                print(f"  警告：找不到 {filepath}，跳过")
                continue
            with open(filepath, "rb") as f:
                content = base64.b64encode(f.read()).decode()
            api_url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{GITHUB_IMAGE_DIR}/{filename}"
            # 检查文件是否已存在（避免重复上传报错）
            check = req.get(api_url, headers=headers, timeout=30)
            body = {"message": f"add survey image {filename}", "content": content, "branch": GITHUB_BRANCH}
            if check.status_code == 200:
                body["sha"] = check.json()["sha"]
            resp = req.put(api_url, headers=headers, json=body, timeout=60)
            if resp.status_code not in (200, 201):
                raise RuntimeError(f"GitHub 上传失败 {filename}: {resp.text[:200]}")
            raw_url = f"https://raw.githubusercontent.com/{GITHUB_REPO}/{GITHUB_BRANCH}/{GITHUB_IMAGE_DIR}/{filename}"
            file_urls[filename] = raw_url
            print(f"  上传完成：{filename}")
    return file_urls


def build_form_requests(file_urls):
    """构造 Forms API batchUpdate requests 列表"""
    requests = []
    insert_index = 0

    description = (
        "感谢参与本研究！\n\n"
        "本问卷共 10 道题，每题约 1-2 分钟。\n\n"
        "操作方式：\n"
        "1. 认真阅读日记内容\n"
        "2. 查看 3 张对应的像素图（A / B / C）\n"
        "3. 选择你认为最能表达这篇日记情绪的那一张\n\n"
        "没有标准答案，请根据直觉和感受作答。图片是用 AI 根据日记生成的像素艺术风格插图。"
    )

    # 表单说明段落
    requests.append({
        "createItem": {
            "item": {
                "title": "问卷说明",
                "description": description,
                "textItem": {},
            },
            "location": {"index": insert_index},
        }
    })
    insert_index += 1

    for diary_num in range(1, 11):
        title, text = DIARIES[diary_num]

        # 日记文字段落
        requests.append({
            "createItem": {
                "item": {
                    "title": title,
                    "description": text,
                    "textItem": {},
                },
                "location": {"index": insert_index},
            }
        })
        insert_index += 1

        # 3 张图片
        for letter in ["A", "B", "C"]:
            filename = f"{diary_num}{letter}.png"
            image_uri = file_urls.get(filename)
            if not image_uri:
                print(f"  警告：{filename} 没有 URL，跳过图片")
                continue
            requests.append({
                "createItem": {
                    "item": {
                        "title": f"图 {letter}",
                        "imageItem": {
                            "image": {
                                "sourceUri": image_uri,
                                "altText": f"diary {diary_num} 图{letter}",
                                "properties": {"alignment": "LEFT", "width": 400},
                            }
                        },
                    },
                    "location": {"index": insert_index},
                }
            })
            insert_index += 1

        # 单选题
        requests.append({
            "createItem": {
                "item": {
                    "title": "哪张图最能表达这篇日记的情绪？",
                    "questionItem": {
                        "question": {
                            "required": True,
                            "choiceQuestion": {
                                "type": "RADIO",
                                "options": [
                                    {"value": "A"},
                                    {"value": "B"},
                                    {"value": "C"},
                                ],
                            },
                        }
                    },
                },
                "location": {"index": insert_index},
            }
        })
        insert_index += 1

    return requests


def main():
    if not os.path.exists(SA_FILE):
        print(f"错误：找不到 {SA_FILE}")
        print("请将 service_account.json 放到项目根目录。")
        return

    print("=== 开始授权 ===")
    creds = authenticate()
    forms_service = build("forms", "v1", credentials=creds)

    urls_path = os.path.join(OUTPUTS_DIR, "imgur_urls.json")
    if os.path.exists(urls_path):
        print("\n=== 使用已缓存的图片 URL ===")
        with open(urls_path) as f:
            file_urls = json.load(f)
        print(f"已加载 {len(file_urls)} 张图片 URL")
    else:
        with open(TOKEN_FILE) as f:
            github_token = f.read().strip()
        print("\n=== 上传图片到 GitHub ===")
        file_urls = upload_images_to_github(github_token)
        print(f"共上传 {len(file_urls)} 张图片")
        with open(urls_path, "w", encoding="utf-8") as f:
            json.dump(file_urls, f, ensure_ascii=False, indent=2)
        print(f"图片 URL 已保存到 {urls_path}")

    print("\n=== 创建 Google Form ===")
    form_body = {
        "info": {
            "title": "情绪像素图偏好测试",
            "documentTitle": "emotion-pixel-viz-survey",
        }
    }
    form = forms_service.forms().create(body=form_body).execute()
    form_id = form["formId"]
    form_url = f"https://docs.google.com/forms/d/{form_id}/edit"
    print(f"表单已创建：{form_url}")

    print("\n=== 添加题目 ===")
    requests = build_form_requests(file_urls)
    forms_service.forms().batchUpdate(
        formId=form_id,
        body={"requests": requests},
    ).execute()
    print(f"共添加 {len(requests)} 个 items")

    respond_url = f"https://docs.google.com/forms/d/{form_id}/viewform"
    print(f"\n=== 完成 ===")
    print(f"编辑链接：{form_url}")
    print(f"填写链接：{respond_url}")

    result_path = os.path.join(OUTPUTS_DIR, "form_links.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump({"edit_url": form_url, "respond_url": respond_url, "form_id": form_id}, f, indent=2)
    print(f"链接已保存到 {result_path}")


if __name__ == "__main__":
    main()
