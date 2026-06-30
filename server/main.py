import os
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
    image: str          # base64 encoded image (with data:image/... prefix)
    prompt: str         # user's editing instruction


@app.post("/api/edit")
async def edit_image(req: EditRequest):
    if not SF_API_KEY:
        raise HTTPException(500, "SF_API_KEY not configured")

    headers = {
        "Authorization": f"Bearer {SF_API_KEY}",
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient(timeout=180.0) as client:
        payload = {
            "model": SF_MODEL,
            "prompt": req.prompt,
            "image": req.image,
            "image_size": "1024x1024"
        }

        resp = await client.post(
            SF_BASE_URL,
            headers=headers,
            json=payload
        )
        data = resp.json()

        if resp.status_code != 200:
            error_msg = data.get("message", str(data))
            raise HTTPException(500, f"SiliconFlow error ({resp.status_code}): {error_msg}")

        # 获取生成的图片 URL
        images = data.get("images", [])
        if not images:
            raise HTTPException(500, f"No image in response: {str(data)[:500]}")

        image_url = images[0].get("url", "")
        if not image_url:
            raise HTTPException(500, f"Empty image URL in response: {str(data)[:500]}")

        # 下载生成的图片并转 base64
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


# Serve static frontend
app.mount("/", StaticFiles(directory="static", html=True), name="static")
