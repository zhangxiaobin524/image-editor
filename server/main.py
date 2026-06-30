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
SF_IMAGE_URL = "https://api.siliconflow.cn/v1/images/generations"
SF_MODEL = "Qwen/Qwen-Image-Edit-2509"


class EditRequest(BaseModel):
    image: str       # base64 ID card image
    new_id: str      # new 18-digit ID number


def parse_birth_from_id(id_num: str) -> str:
    """从身份证号提取出生日期，如 1995年5月24日"""
    y = id_num[6:10]
    m = id_num[10:12]
    d = id_num[12:14]
    return f"{y}年{int(m)}月{int(d)}日"


def build_prompt(new_id: str) -> str:
    """修改身份证号 + 同步修改出生日期"""
    birth = parse_birth_from_id(new_id)
    return (
        f"身份证有两个地方需要修改（必须同时修改）：\n"
        f"1. 公民身份号码：改为 {new_id}\n"
        f"2. 出生日期：改为 {birth}\n\n"
        "新文字的颜色必须和原身份证上其他印刷文字（如姓名、性别、民族、住址等）完全相同——深蓝灰色，不是纯黑。\n"
        "字体大小、粗细、间距必须和原文字完全一致。\n"
        "除了这两处（号码和出生日期），身份证上其他所有内容（姓名、性别、民族、住址、照片、签发机关、有效期、背景、边框、水印）保持100%完全不变。\n"
        "不要改变图片任何其他部分。"
    )


@app.post("/api/edit")
async def edit_image(req: EditRequest):
    if not SF_API_KEY:
        raise HTTPException(500, "SF_API_KEY not configured")

    prompt = build_prompt(req.new_id)

    headers = {
        "Authorization": f"Bearer {SF_API_KEY}",
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient(timeout=180.0) as client:
        payload = {
            "model": SF_MODEL,
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
            raise HTTPException(500, "No image in response")

        result_url = images[0].get("url", "")
        dl_resp = await client.get(result_url)
        if dl_resp.status_code != 200:
            raise HTTPException(500, f"Download failed: {dl_resp.status_code}")

        ct = dl_resp.headers.get("content-type", "image/png")
        img_b64 = base64.b64encode(dl_resp.content).decode()

        return {
            "code": 0,
            "data": {"image": f"data:{ct};base64,{img_b64}"}
        }


@app.get("/api/health")
def health():
    return {"status": "ok", "key_configured": bool(SF_API_KEY)}

app.mount("/", StaticFiles(directory="static", html=True), name="static")
