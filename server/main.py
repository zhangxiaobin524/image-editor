import os
import io
import base64
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from PIL import Image, ImageFilter

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
    """Extract birth date from 18-digit ID number."""
    year = id_number[6:10]
    month = str(int(id_number[10:12]))   # remove leading zero
    day = str(int(id_number[12:14]))     # remove leading zero
    return f"{year}年{month}月{day}日"


def decode_base64_image(b64_str: str) -> Image.Image:
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    data = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(data))


def encode_image_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"


def composite_photo(idcard_b64: str, photo_b64: str) -> str:
    """Paste new avatar onto ID card with feathered edges for natural blending."""
    idcard_img = decode_base64_image(idcard_b64).convert("RGBA")
    photo_img = decode_base64_image(photo_b64).convert("RGBA")

    w, h = idcard_img.size

    # ID card photo area
    pw = int(w * 0.20)
    ph = int(h * 0.30)
    px = int(w * 0.68)
    py = int(h * 0.10)

    # Center-crop photo to target aspect ratio
    photo_ratio = photo_img.width / photo_img.height
    target_ratio = pw / ph

    if photo_ratio > target_ratio:
        new_w = int(photo_img.height * target_ratio)
        left = (photo_img.width - new_w) // 2
        photo_img = photo_img.crop((left, 0, left + new_w, photo_img.height))
    else:
        new_h = int(photo_img.width / target_ratio)
        top = (photo_img.height - new_h) // 2
        photo_img = photo_img.crop((0, top, photo_img.width, top + new_h))

    photo_resized = photo_img.resize((pw, ph), Image.LANCZOS)

    # Create feathered mask: white center fading to transparent at edges
    mask = Image.new("L", (pw, ph), 255)
    feather = max(pw, ph) // 10  # 10% feather radius
    if feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=feather))

    # Paste with feathered mask
    idcard_img.paste(photo_resized, (px, py), mask)

    return encode_image_to_base64(idcard_img.convert("RGB"))


def build_prompt(new_id: str, birth_date: str, new_name: str, new_address: str, has_photo: bool) -> str:
    """Build editing prompt. Birth date is ALWAYS included (derived from ID number)."""
    lines = []

    # --- Must-change items ---
    lines.append(f"1. 将公民身份号码改为：{new_id}")
    lines.append(f"2. 将出生日期改为：{birth_date}")

    idx = 3

    if new_name:
        lines.append(f"{idx}. 将姓名改为：{new_name}")
        idx += 1

    if new_address:
        lines.append(f"{idx}. 将住址改为：{new_address}")
        idx += 1

    if has_photo:
        lines.append(f"{idx}. 身份证右上角已放置新头像照片（有羽化过渡边缘），请将新头像与身份证自然融合：调整头像的亮度、对比度、色调、颗粒感，使其与身份证整体风格完全一致，就像原本就是印在这张身份证上的一样。头像内容保持不变。")

    # Build strict requirements section
    reqs = [
        "【关键】身份证号码和出生日期的文字颜色必须观察原始身份证文字的实际颜色来匹配。身份证上的印刷文字是深蓝灰色或深灰色，不是纯黑色(#000000)。请仔细观察原来字号、住址等文字的颜色深浅，新写的号码和日期颜色必须与它们完全一样，深浅浓淡一致。",
        "只修改上述列出的项目，身份证上其他所有内容（性别、民族、签发机关、有效期限、防伪水印、底纹等）保持100%不变",
        "所有文字的字体、大小、粗细、位置、字符间距与原始身份证完全一致",
        "不要改变图片整体的亮度、对比度、背景色、边框",
    ]

    prompt = "请对这张身份证进行以下修改：\n" + "\n".join(lines)
    prompt += "\n\n严格要求：\n" + "\n".join(f"- {r}" for r in reqs)
    return prompt


@app.post("/api/edit")
async def edit_image(req: EditRequest):
    if not SF_API_KEY:
        raise HTTPException(500, "SF_API_KEY not configured")

    birth_date = extract_birth_date(req.new_id)
    has_photo = bool(req.new_photo)

    if has_photo:
        try:
            work_image = composite_photo(req.image, req.new_photo)
        except Exception as e:
            raise HTTPException(400, f"头像合成失败: {str(e)}")
    else:
        work_image = req.image

    prompt = build_prompt(req.new_id, birth_date, req.new_name, req.new_address, has_photo)

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
        img_b64 = base64.b64encode(dl_resp.content).decode("utf-8")

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
