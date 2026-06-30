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

# Try multiple vision models in order of preference
VISION_MODELS = [
    "Qwen/Qwen2-VL-72B-Instruct",
    "Qwen/Qwen2-VL-7B-Instruct",
    "deepseek-ai/deepseek-vl2",
]


class EditRequest(BaseModel):
    image: str          # base64 ID card image
    new_id: str         # new 18-digit ID number
    new_name: str = ""  # optional: new name
    new_address: str = ""  # optional: new address
    new_photo: str = ""   # reserved


def extract_birth_date(id_number: str) -> str:
    year = id_number[6:10]
    month = str(int(id_number[10:12]))
    day = str(int(id_number[12:14]))
    return f"{year}年{month}月{day}日"


# ========== Vision OCR (with multi-model fallback) ==========

IDCARD_OCR_PROMPT = """你是一个身份证OCR识别器。请仔细阅读这张中国居民身份证图片，提取以下字段并返回严格的JSON格式（不要markdown代码块，只要纯JSON）。

逐字核对确保准确。公民身份号码必须完整18位。住址完整不省略。看不清的填"未知"。
返回JSON：
{"姓名":"...","性别":"...","民族":"...","出生":"...","住址":"...","公民身份号码":"...","签发机关":"...","有效期限":"..."}"""


async def ocr_id_card(client: httpx.AsyncClient, image_b64: str) -> dict:
    """Try each vision model until one works."""
    headers = {
        "Authorization": f"Bearer {SF_API_KEY}",
        "Content-Type": "application/json"
    }

    payload_base = {
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_b64}},
                {"type": "text", "text": IDCARD_OCR_PROMPT}
            ]
        }],
        "temperature": 0.1,
        "max_tokens": 1024
    }

    last_error = None
    for model in VISION_MODELS:
        payload = {**payload_base, "model": model}
        try:
            resp = await client.post(SF_CHAT_URL, headers=headers, json=payload, timeout=60.0)
            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"]["content"]
                return _parse_ocr_json(content)
            else:
                last_error = f"{model}: {resp.status_code} - {resp.text[:200]}"
        except Exception as e:
            last_error = f"{model}: {e}"

    # All models failed — return None so caller uses template fallback
    return None


def _parse_ocr_json(content: str) -> dict:
    content = content.strip()
    if content.startswith("```"):
        lines = content.split("\n")
        content = "\n".join(lines[1:-1])
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        import re
        match = re.search(r'\{[\s\S]*\}', content)
        if match:
            return json.loads(match.group())
        raise ValueError(f"Bad OCR JSON: {content[:300]}")


# ========== Prompt builders ==========

def build_prompt_with_ocr(original: dict, new_id: str, birth_date: str,
                          new_name: str, new_address: str) -> str:
    """Build prompt using OCR text as exact anchors."""
    orig_num = original.get("公民身份号码", "")
    orig_birth = original.get("出生", "")
    orig_name = original.get("姓名", "")
    orig_addr = original.get("住址", "")

    p = []
    p.append("精确替换身份证文字，只改指定位置的文字，其他100%不变：")
    p.append("")
    p.append(f'1. 找到号码「{orig_num}」，改为「{new_id}」。颜色与原身份证文字一致（深蓝灰），禁止纯黑。')
    p.append(f'2. 找到出生日期「{orig_birth}」，改为「{birth_date}」。（来自新号码第7~14位）')
    if new_name and new_name != orig_name:
        p.append(f'3. 找到姓名「{orig_name}」，改为「{new_name}」。颜色同上。')
    if new_addr and new_addr != orig_addr:
        p.append(f'4. 找到住址「{orig_addr}」，改为「{new_addr}」。')
    p.append("")
    p.append("【绝对禁止】修改性别、民族、签发机关、有效期限、头像、背景、边框、水印、亮度对比度")
    return "\n".join(p)


def build_template_prompt(new_id: str, birth_date: str, new_name: str, new_address: str) -> str:
    """Build prompt without OCR — describe standard Chinese ID card layout."""
    p = []
    p.append("这是一张中国居民二代身份证照片。请精确修改以下字段（标准布局参考）：")
    p.append("")
    p.append(f'1.【公民身份号码】位于卡片底部一行，标签为"公民身份号码"，后面跟着18位数字/字母。将原号码完整替换为「{new_id}」（18位）。新号码的颜色必须是深蓝灰色（和姓名、住址等印刷字同色），严禁纯黑色(#000000)，字体大小粗细与原号码完全一致。')
    p.append('')
    p.append(f'2.【出生】位于左侧第三行，标签为"出生"，后面是出生日期如"1993年5月24日"。将其改为「{birth_date}」。这是从新身份证号第7~14位({new_id[6:10]}{new_id[10:12]}{new_id[12:14]})推导出来的，必须同步改！')
    if new_name:
        p.append(f'3.【姓名】位于左侧第一行，标签为"姓名"。将原姓名替换为「{new_name}」。')
    if new_address:
        p.append(f'4.【住址】位于左侧第四行起，标签为"住址"。将原住址替换为「{new_address}」。')
    p.append('')
    p.append('【绝对禁止】修改未列出的任何内容——性别、民族、头像照片、签发机关、有效期限、背景色、边框、防伪纹路、水印都保持100%原样。不要改变整图亮度对比度。所有新写文字统一深蓝灰色，禁止纯黑。')
    return "\n".join(p)


# ========== API endpoint ==========

@app.post("/api/edit")
async def edit_image(req: EditRequest):
    if not SF_API_KEY:
        raise HTTPException(500, "SF_API_KEY not configured")

    birth_date = extract_birth_date(req.new_id)
    original = None
    used_template = False
    prompt = ""

    async with httpx.AsyncClient(timeout=180.0) as client:

        # Step 1: Try OCR (optional — fail gracefully)
        try:
            original = await ocr_id_card(client, req.image)
        except Exception as e:
            pass  # Will use template fallback

        # Step 2: Build prompt
        if original and isinstance(original, dict):
            prompt = build_prompt_with_ocr(original, req.new_id, birth_date,
                                          req.new_name, req.new_address)
        else:
            used_template = True
            prompt = build_template_prompt(req.new_id, birth_date,
                                         req.new_name, req.new_address)

        # Step 3: Image edit
        headers = {
            "Authorization": f"Bearer {SF_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": SF_EDIT_MODEL,
            "prompt": prompt,
            "image": req.image,
            "image_size": "1024x1024"
        }

        resp = await client.post(SF_IMAGE_URL, headers=headers, json=payload)
        data = resp.json()

        if resp.status_code != 200:
            raise HTTPException(500, f"Edit failed ({resp.status_code}): {data.get('message', '')[:300]}")

        images = data.get("images", [])
        if not images:
            raise HTTPException(500, f"No image in response")

        result_url = images[0].get("url", "")
        dl_resp = await client.get(result_url)
        if dl_resp.status_code != 200:
            raise HTTPException(500, f"Download failed: {dl_resp.status_code}")

        ct = dl_resp.headers.get("content-type", "image/png")
        img_b64 = base64.b64encode(dl_resp.content).decode()

        result_data = {
            "image": f"data:{ct};base64,{img_b64}",
            "prompt": prompt,
            "ocr_used": not used_template,
        }
        if original:
            result_data["original"] = original

        return {"code": 0, "data": result_data}


@app.get("/api/health")
def health():
    return {"status": "ok", "key_configured": bool(SF_API_KEY)}

app.mount("/", StaticFiles(directory="static", html=True), name="static")
