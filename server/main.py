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
SF_URL = "https://api.siliconflow.cn/v1/images/generations"
MODEL = "Qwen/Qwen-Image-Edit-2509"


class EditRequest(BaseModel):
    image: str
    new_id: str


def parse_birth(id_num: str) -> str:
    y, m, d = id_num[6:10], id_num[10:12], id_num[12:14]
    return f"{y}年{int(m)}月{int(d)}日"


async def call_ai(client: httpx.AsyncClient, image: str, prompt: str):
    """Call Qwen-Image-Edit once, return result image URL."""
    resp = await client.post(
        SF_URL,
        headers={"Authorization": f"Bearer {SF_API_KEY}", "Content-Type": "application/json"},
        json={"model": MODEL, "prompt": prompt, "image": image, "num_images": 1, "seed": 42},
    )
    if resp.status_code != 200:
        d = resp.json() if resp.text else {}
        raise Exception(f"{resp.status_code}: {d.get('message', '')[:200]}")
    imgs = resp.json().get("images", [])
    return (imgs[0].get("url") or imgs[0]) if imgs else None


async def url_to_base64(client: httpx.AsyncClient, url: str) -> str:
    """Download image URL and return as data:image/png;base64,..."""
    r = await client.get(url)
    r.raise_for_status()
    b64 = base64.b64encode(r.content).decode()
    return f"data:image/png;base64,{b64}"


@app.post("/api/edit")
async def edit_image(req: EditRequest):
    birth = parse_birth(req.new_id)
    if not SF_API_KEY:
        raise HTTPException(500, "API Key 未配置")

    async with httpx.AsyncClient(timeout=120) as c:
        # Step 1: 改号码 (input: base64)
        img_step1 = await call_ai(c, req.image,
            f"把身份证底部的公民身份号码精确替换为{req.new_id}。只改这18位数字，其他所有内容100%保持不变，字体颜色和原来一模一样。")
        if not img_step1:
            raise HTTPException(500, "Step1 AI未返回图片")

        # Step1 返回的是URL，需下载转base64才能传给Step2
        img_step1_b64 = await url_to_base64(c, img_step1)

        # Step 2: 改出生日期 (input: base64 from Step1)
        img_final = await call_ai(c, img_step1_b64,
            f"把身份证上的「出生」那一行日期完整替换为{birth}（年月日三个数字全部改）。只改这一处文字，其他内容100%保持不变，字体颜色和原证件上其他印刷文字完全一致。")
        if not img_final:
            raise HTTPException(500, "Step2 AI未返回图片")

    return {"code": 0, "data": {
        "image": img_final,
        "birth_date": birth,
        "new_id": req.new_id,
        "steps": ["改号码→" + req.new_id, "改出生→" + birth],
    }}


@app.get("/api/health")
def health():
    return {"status": "ok", "api_configured": bool(SF_API_KEY)}


app.mount("/", StaticFiles(directory="static", html=True), name="static")
