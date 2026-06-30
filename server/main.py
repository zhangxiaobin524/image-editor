import os
import io
import base64
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from PIL import Image

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
    """Extract birth date from 18-digit ID number: YYYY年MM月DD日"""
    year = id_number[6:10]
    month = id_number[10:12]
    day = id_number[12:14]
    return f"{year}年{month}月{day}日"


def decode_base64_image(b64_str: str) -> Image.Image:
    """Decode base64 (with or without data: prefix) to PIL Image."""
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    data = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(data))


def encode_image_to_base64(img: Image.Image) -> str:
    """Encode PIL Image to base64 data URL."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"


def composite_photo(idcard_b64: str, photo_b64: str) -> str:
    """Paste new avatar photo onto the ID card image at the photo position.
    Returns composited image as base64 data URL."""
    idcard_img = decode_base64_image(idcard_b64)
    photo_img = decode_base64_image(photo_b64)

    w, h = idcard_img.size

    # ID card photo area: top-right, ~20% width, ~30% height
    pw = int(w * 0.20)
    ph = int(h * 0.30)
    px = int(w * 0.68)  # 68% from left
    py = int(h * 0.10)  # 10% from top

    # Resize photo to fit, crop to fill (center crop)
    photo_ratio = photo_img.width / photo_img.height
    target_ratio = pw / ph

    if photo_ratio > target_ratio:
        # Photo is wider - crop width
        new_w = int(photo_img.height * target_ratio)
        left = (photo_img.width - new_w) // 2
        photo_img = photo_img.crop((left, 0, left + new_w, photo_img.height))
    else:
        # Photo is taller - crop height
        new_h = int(photo_img.width / target_ratio)
        top = (photo_img.height - new_h) // 2
        photo_img = photo_img.crop((0, top, photo_img.width, top + new_h))

    photo_resized = photo_img.resize((pw, ph), Image.LANCZOS)

    # Paste with alpha support
    if photo_resized.mode == 'RGBA':
        idcard_img.paste(photo_resized, (px, py), photo_resized)
    else:
        idcard_img.paste(photo_resized, (px, py))

    return encode_image_to_base64(idcard_img)


def build_prompt(new_id: str, birth_date: str, new_name: str, new_address: str, has_photo: bool) -> str:
    """Build a comprehensive editing prompt based on all fields."""
    changes = []
    changes.append(f"修改公民身份号码为：{new_id}")
    changes.append(f"修改出生日期为：{birth_date}")

    if new_name:
        changes.append(f"修改姓名为：{new_name}")

    if new_address:
        changes.append(f"修改住址为：{new_address}")

    if has_photo:
        changes.append("身份证右上角照片区域已有新头像，请保持该头像内容不变，仅调整其光照和色调，使其与身份证背景自然融合，看起来像原本就印在上面一样")

    change_list = "\n".join([f"  {i+1}) {c}" for i, c in enumerate(changes)])

    prompt = f"""请对这张身份证进行以下修改：
{change_list}

严格要求：
- 只修改上述指定内容，身份证上其他所有信息（性别、民族、签发机关、有效期限、防伪水印等）保持完全不变
- 所有文字的颜色、字体、大小、粗细、位置和字符间距必须与原始身份证完全一致，不准使用纯黑色
- 修改后的文字看起来就像原本打印在上面的一样，自然无痕
- 不要改变图片的亮度、对比度、背景、边框等任何其他部分"""

    return prompt


@app.post("/api/edit")
async def edit_image(req: EditRequest):
    if not SF_API_KEY:
        raise HTTPException(500, "SF_API_KEY not configured")

    # Extract birth date from ID number
    birth_date = extract_birth_date(req.new_id)

    # If new photo provided, composite it onto the ID card
    has_photo = bool(req.new_photo)
    if has_photo:
        try:
            work_image = composite_photo(req.image, req.new_photo)
        except Exception as e:
            raise HTTPException(400, f"头像合成失败: {str(e)}")
    else:
        work_image = req.image

    # Build prompt
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
