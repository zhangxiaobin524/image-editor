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
    y = id_num[6:10]
    m = id_num[10:12]
    d = id_num[12:14]
    return f"{y}年{int(m)}月{int(d)}日"


def build_prompt(new_id: str) -> str:
    birth = parse_birth_from_id(new_id)
    # 简短精确的指令，避免模型自由发挥
    return (
        f"把身份证上的公民身份号码改成 {new_id}，"
        f"出生日期改成 {birth}。"
        f"只改这两个地方，其他所有文字、照片、背景完全保持原样不动。"
        f"新数字的颜色和字体要和原来的号码一模一样。"
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
            # 注意：Qwen-Image-Edit 不支持 image_size 字段！
            # 也支持 image2/image3 可传入参考图
            "num_inference_steps": 28,
            "seed": 42,
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
            "data": {
                "image": f"data:{ct};base64,{img_b64}",
                "prompt": prompt,
                "birth_date": parse_birth_from_id(req.new_id),
            }
        }


@app.get("/api/health")
def health():
    return {"status": "ok", "key_configured": bool(SF_API_KEY)}

app.mount("/", StaticFiles(directory="static", html=True), name="static")
