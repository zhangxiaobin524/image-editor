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


@app.post("/api/edit")
async def edit_image(req: EditRequest):
    birth = parse_birth(req.new_id)

    if not SF_API_KEY:
        raise HTTPException(500, "SiliconFlow API Key 未配置")

    prompt = (
        f"身份证修改任务：把公民身份号码改为{req.new_id}，"
        f"把出生日期改为{birth}。"
        f"只改这两处，其他内容完全不动。"
    )

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            SF_URL,
            headers={
                "Authorization": f"Bearer {SF_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL,
                "prompt": prompt,
                "image": req.image,
                "num_images": 1,
                "seed": 42,
            },
        )

    if resp.status_code != 200:
        data = resp.json() if resp.text else {}
        msg = data.get("message", str(resp.status_code))
        raise HTTPException(resp.status_code, f"AI调用失败: {msg[:200]}")

    data = resp.json()
    images = data.get("images", []) if isinstance(data, dict) else data
    if not images or not isinstance(images, list):
        raise HTTPException(500, "AI未返回图片")

    return {
        "code": 0,
        "data": {
            "image": images[0].get("url") or images[0],
            "birth_date": birth,
            "new_id": req.new_id,
        }
    }


@app.get("/api/health")
def health():
    return {"status": "ok", "api_configured": bool(SF_API_KEY)}


app.mount("/", StaticFiles(directory="static", html=True), name="static")
