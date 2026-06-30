import os
import io
import base64
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

SF_API_KEY = os.getenv("SF_API_KEY", "")
SF_BASE_URL = "https://api.siliconflow.cn/v1/images/generations"
SF_MODEL = "Qwen/Qwen-Image-Edit-2509"


class EditRequest(BaseModel):
    image: str          # base64 ID card image
    new_id: str         # new 18-digit ID number
    new_name: str = ""  # optional: new name
    new_address: str = ""  # optional: new address
    new_photo: str = ""   # optional: base64 new avatar photo


def extract_birth_date(id_number: str) -> str:
    """从18位身份证号提取出生日期，格式匹配真实身份证：1993年5月24日"""
    year = id_number[6:10]
    month = str(int(id_number[10:12]))   # 去前导零：05 → 5
    day = str(int(id_number[12:14]))     # 去前导零：24 → 24
    return f"{year}年{month}月{day}日"


def build_prompt(new_id: str, birth_date: str, new_name: str, new_address: str) -> str:
    """
    极其精确的 prompt：每个字段都给出具体值和格式示例。
    不再使用编号列表（模型容易忽略后面的项），而是用自然语言逐条强调。
    """
    # 收集要修改的项
    changes = []

    # 身份证号 —— 必须修改
    changes.append(
        f"【公民身份号码】必须改为 {new_id}（共18位，最后一位是{new_id[-1]}）。"
        f"颜色要求：仔细观察原身份证上「姓名」「住址」等文字的颜色深浅——它们不是纯黑色而是深蓝灰色。"
        f"新号码的每一个数字都必须使用与姓名、住址文字完全相同的颜色、粗细、字体大小。"
        f"绝对禁止使用纯黑色(#000000)。"
    )

    # 出生日期 —— 必须跟随身份证号一起改
    changes.append(
        f"【出生】必须改为 {birth_date}。"
        f"格式必须是「{birth_date}」这样的中文格式，数字之间用「年」「月」「日」分隔。"
        f"这个日期来自新身份证号第7-14位({new_id[6:10]}{new_id[10:12]}{new_id[12:14]})，必须同步修改！"
    )

    # 姓名
    if new_name:
        changes.append(f"【姓名】必须改为「{new_name}」")

    # 住址
    if new_address:
        changes.append(f"【住址】必须改为「{new_address}」")

    # 组装 prompt
    prompt_parts = []
    prompt_parts.append("这是一张中国居民身份证照片。你需要对其中的文字信息进行精确修改。")
    prompt_parts.append("")
    prompt_parts.append("需要修改的内容（每一项都必须执行，不能遗漏）：")

    for i, change in enumerate(changes, 1):
        prompt_parts.append(f"{i}. {change}")

    prompt_parts.append("")
    prompt_parts.append("绝对不能修改的内容（保持100%不变）：")
    prompt_parts.append("- 性别、民族")
    prompt_parts.append("- 签发机关、有效期限")
    prompt_parts.append("- 身份证的整体布局、背景色、边框、防伪纹路、水印")
    prompt_parts.append("- 如果有头像照片，保持头像不变")
    prompt_parts.append("- 图片整体的亮度、对比度都不要改变")
    prompt_parts.append("")
    prompt_parts.append("文字排版要求：")
    prompt_parts.append("- 新写入的文字字体、字号、字间距必须和原来该位置的文字完全一致")
    prompt_parts.append("- 文字位置必须精准对齐到原来的对应位置")
    prompt_parts.append("- 所有新文字的颜色统一为深蓝灰色（与原有姓名/住址文字同色），严禁纯黑")

    return "\n".join(prompt_parts)


@app.post("/api/edit")
async def edit_image(req: EditRequest):
    if not SF_API_KEY:
        raise HTTPException(500, "SF_API_KEY not configured")

    birth_date = extract_birth_date(req.new_id)

    # 直接使用原始身份证图片，不做任何预处理
    work_image = req.image

    prompt = build_prompt(req.new_id, birth_date, req.new_name, req.new_address)

    headers = {
        "Authorization": f"Bearer {SF_API_KEY}",
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient(timeout=180.0) as client:
        payload = {
            "model": SF_MODEL,
            "prompt": prompt,
            "image": work_image,
            "image_size": "1024x1024"
        }

        resp = await client.post(SF_BASE_URL, headers=headers, json=payload)
        data = resp.json()

        if resp.status_code != 200:
            error_msg = data.get("message", str(data))
            raise HTTPException(500, f"SiliconFlow error ({resp.status_code}): {error_msg}")

        images = data.get("images", [])
        if not images:
            raise HTTPException(500, f"No image in response: {str(data)[:500]}")

        image_url = images[0].get("url", "")
        if not image_url:
            raise HTTPException(500, f"Empty image URL in response: {str(data)[:500]}")

        dl_resp = await client.get(image_url)
        if dl_resp.status_code != 200:
            raise HTTPException(500, f"Failed to download result image: {dl_resp.status_code}")

        content_type = dl_resp.headers.get("content-type", "image/png")
        img_b64 = base64.b64encode(dl_resp.content).decode('utf-8')

        return {
            "code": 0,
            "data": {
                "image": f"data:{content_type};base64,{img_b64}"
            }
        }


@app.get("/api/health")
def health():
    return {"status": "ok", "key_configured": bool(SF_API_KEY)}


app.mount("/", StaticFiles(directory="static", html=True), name="static")
