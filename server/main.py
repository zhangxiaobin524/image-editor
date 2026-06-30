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


async def call_edit(client, prompt: str, image_b64: str) -> bytes:
    """Call Qwen Image Edit API once, return raw image bytes."""
    payload = {
        "model": SF_MODEL,
        "prompt": prompt,
        "image": image_b64,
        "num_inference_steps": 28,
        "seed": 42,
    }
    resp = await client.post(SF_IMAGE_URL, json=payload)
    data = resp.json()

    if resp.status_code != 200:
        raise Exception(f"API error ({resp.status_code}): {data.get('message', '')[:300]}")

    images = data.get("images", [])
    if not images:
        raise Exception("No image in response")

    result_url = images[0].get("url", "")
    dl_resp = await client.get(result_url)
    if dl_resp.status_code != 200:
        raise Exception(f"Download failed: {dl_resp.status_code}")

    return dl_resp.content


@app.post("/api/edit")
async def edit_image(req: EditRequest):
    if not SF_API_KEY:
        raise HTTPException(500, "SF_API_KEY not configured")

    birth = parse_birth_from_id(req.new_id)

    headers = {
        "Authorization": f"Bearer {SF_API_KEY}",
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient(timeout=300.0) as client:
        client.headers.update(headers)

        # === Step 1: 改号码 ===
        step1_prompt = (
            f"编辑任务：将身份证上的公民身份号码精确替换为 {req.new_id}。"
            f"要求：(1)只修改底部的18位数字，(2)新数字必须使用和原身份证上其他印刷文字完全相同的字体、颜色、粗细、间距，"
            f"(3)身份证上其他所有内容100%保持不变（姓名、性别、民族、出生日期、住址、照片、背景纹路、边框全部不动），"
            f"(4)不要添加任何多余文字或符号。"
        )
        img_bytes_1 = await call_edit(client, step1_prompt, req.image)
        img_b64_1 = base64.b64encode(img_bytes_1).decode()
        ct_1 = "image/png"

        # === Step 2: 改出生日期（基于Step1的结果） ===
        step2_prompt = (
            f"编辑任务：将身份证上的「出生」那一行文字中的日期整行替换。"
            f"找到当前显示的出生日期（格式如 XXXX年X月X日），将其完整替换为 {birth}（包括年、月、日全部替换）。"
            f"要求：(1)年月日三个数字都必须改，不能只改年份，(2)新文字字体、颜色、大小必须和原证件上其他印刷文字完全一致，"
            f"(3)只修改出生日期这一处，其他所有内容100%保持不变。"
        )
        img_bytes_2 = await call_edit(
            client, step2_prompt,
            f"data:{ct_1};base64,{img_b64_1}"   # 用Step1的输出作为输入
        )
        ct_2 = "image/png"

    final_b64 = base64.b64encode(img_bytes_2).decode()

    return {
        "code": 0,
        "data": {
            "image": f"data:{ct_2};base64,{final_b64}",
            "birth_date": birth,
            "steps": [
                {"action": "修改号码", "to": req.new_id},
                {"action": "修改出生日期", "to": birth},
            ]
        }
    }


@app.get("/api/health")
def health():
    return {"status": "ok", "key_configured": bool(SF_API_KEY)}

app.mount("/", StaticFiles(directory="static", html=True), name="static")
