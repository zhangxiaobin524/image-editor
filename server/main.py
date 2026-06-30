import os
import io
import json
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
SF_IMAGE_URL = "https://api.siliconflow.cn/v1/images/generations"
SF_CHAT_URL = "https://api.siliconflow.cn/v1/chat/completions"
SF_EDIT_MODEL = "Qwen/Qwen-Image-Edit-2509"
SF_VISION_MODEL = "Qwen/Qwen2.5-VL-72B-Instruct"


class EditRequest(BaseModel):
    image: str          # base64 ID card image
    new_id: str         # new 18-digit ID number
    new_name: str = ""  # optional: new name
    new_address: str = ""  # optional: new address
    new_photo: str = ""   # optional: base64 new avatar photo (reserved)


# ---------- Utility ----------

def extract_birth_date(id_number: str) -> str:
    """从18位身份证号提取出生日期，格式：1993年5月24日"""
    year = id_number[6:10]
    month = str(int(id_number[10:12]))
    day = str(int(id_number[12:14]))
    return f"{year}年{month}月{day}日"


def strip_data_url(b64: str) -> str:
    if "," in b64:
        return b64.split(",", 1)[1]
    return b64


# ---------- Vision model: OCR the ID card ----------

IDCARD_OCR_PROMPT = """你是一个身份证OCR识别器。请仔细阅读这张中国居民身份证图片，提取以下字段并返回严格的JSON格式（不要markdown代码块，只要纯JSON）。

注意：
- 逐字核对，确保每个字都准确
- 公民身份号码必须完整提取18位
- 住址必须完整提取，不要省略任何字
- 如果某个字段看不清，填"未知"

返回JSON格式如下：
{
  "姓名": "...",
  "性别": "...",
  "民族": "...",
  "出生": "...",
  "住址": "...",
  "公民身份号码": "...",
  "签发机关": "...",
  "有效期限": "..."
}"""


async def ocr_id_card(client: httpx.AsyncClient, image_b64: str) -> dict:
    """Use vision model to read all text fields from the ID card."""
    headers = {
        "Authorization": f"Bearer {SF_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": SF_VISION_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_b64}},
                    {"type": "text", "text": IDCARD_OCR_PROMPT}
                ]
            }
        ],
        "temperature": 0.1,
        "max_tokens": 1024
    }

    resp = await client.post(SF_CHAT_URL, headers=headers, json=payload)

    if resp.status_code != 200:
        raise HTTPException(500, f"OCR failed ({resp.status_code}): {resp.text[:500]}")

    data = resp.json()
    content = data["choices"][0]["message"]["content"]

    # Parse JSON from response (may have markdown fences)
    content = content.strip()
    if content.startswith("```"):
        lines = content.split("\n")
        content = "\n".join(lines[1:-1])

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Try to extract JSON from the text
        import re
        match = re.search(r'\{[\s\S]*\}', content)
        if match:
            return json.loads(match.group())
        raise HTTPException(500, f"OCR returned invalid JSON: {content[:300]}")


# ---------- Build precise edit prompt ----------

def build_prompt(original: dict, new_id: str, birth_date: str, new_name: str, new_address: str) -> str:
    """
    Build prompt using original text as exact anchors.
    Tells the model: "find the text '张三' and change it to '李四'".
    This way the model knows EXACTLY which pixels to modify.
    """
    original_number = original.get("公民身份号码", "未知")
    original_birth = original.get("出生", "未知")
    original_name = original.get("姓名", "未知")
    original_address = original.get("住址", "未知")

    parts = []
    parts.append("你是一个精确的身份证文字编辑器。请在原图上对以下文字进行替换。")
    parts.append("")
    parts.append("【重要】你必须把身份证上的原始文字找出来，替换为新文字。每项替换只改动该处文字，其他区域完全不动。")
    parts.append("")

    # 1. 身份证号
    parts.append(f"1. 找到身份证上原号码「{original_number}」，将它替换为「{new_id}」。")
    parts.append(f"   新号码颜色必须与身份证上原有文字的墨色完全一致（不是纯黑，而是深蓝灰），字体、字号、间距与原文字相同。")

    # 2. 出生日期
    parts.append(f"2. 找到原出生日期「{original_birth}」，将它替换为「{birth_date}」。")
    parts.append(f"   格式是「年」「月」「日」中文分隔，颜色同上。")

    # 3. 姓名
    if new_name and new_name != original_name:
        parts.append(f"3. 找到原姓名「{original_name}」，将它替换为「{new_name}」。颜色、字体同上。")

    # 4. 住址
    if new_address and new_address != original_address:
        parts.append(f"4. 找到原住址「{original_address}」，将它替换为「{new_address}」。颜色、字体同上。")

    parts.append("")
    parts.append("【绝对禁止】")
    parts.append("- 禁止修改未列出的任何文字（如性别、民族、签发机关、有效期限）")
    parts.append("- 禁止修改头像照片、背景、边框、水印、防伪纹路")
    parts.append("- 禁止改变整图的亮度、对比度、色调")
    parts.append("- 禁止在任何不该有文字的地方写新文字")
    parts.append("- 禁止使用纯黑色(#000000)，所有文字必须是深蓝灰色")

    return "\n".join(parts)


# ---------- API endpoint ----------

@app.post("/api/edit")
async def edit_image(req: EditRequest):
    if not SF_API_KEY:
        raise HTTPException(500, "SF_API_KEY not configured")

    birth_date = extract_birth_date(req.new_id)

    async with httpx.AsyncClient(timeout=180.0) as client:

        # Step 1: OCR the ID card to get original text
        image_url = req.image
        original = await ocr_id_card(client, image_url)

        # Step 2: Build precise prompt with original text anchors
        prompt = build_prompt(original, req.new_id, birth_date, req.new_name, req.new_address)

        # Step 3: Call image edit model
        headers = {
            "Authorization": f"Bearer {SF_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": SF_EDIT_MODEL,
            "prompt": prompt,
            "image": image_url,
            "image_size": "1024x1024"
        }

        resp = await client.post(SF_IMAGE_URL, headers=headers, json=payload)
        data = resp.json()

        if resp.status_code != 200:
            error_msg = data.get("message", str(data))
            raise HTTPException(500, f"Image edit failed ({resp.status_code}): {error_msg}")

        images = data.get("images", [])
        if not images:
            raise HTTPException(500, f"No image in response: {str(data)[:500]}")

        result_url = images[0].get("url", "")
        if not result_url:
            raise HTTPException(500, f"Empty image URL in response: {str(data)[:500]}")

        dl_resp = await client.get(result_url)
        if dl_resp.status_code != 200:
            raise HTTPException(500, f"Failed to download result image: {dl_resp.status_code}")

        content_type = dl_resp.headers.get("content-type", "image/png")
        img_b64 = base64.b64encode(dl_resp.content).decode('utf-8')

        return {
            "code": 0,
            "data": {
                "image": f"data:{content_type};base64,{img_b64}",
                "original": original,   # OCR result for debugging
                "prompt": prompt        # the actual edit prompt used
            }
        }


@app.get("/api/health")
def health():
    return {"status": "ok", "key_configured": bool(SF_API_KEY)}


app.mount("/", StaticFiles(directory="static", html=True), name="static")
