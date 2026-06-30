import os
import base64
import httpx
import logging
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

VOLC_API_KEY = os.getenv("VOLC_API_KEY", "")
VOLC_URL = "https://ark.cn-beijing.volces.com/api/v3/images/generations"

# 模型候选列表（按优先级尝试）
MODEL_CANDIDATES = [
    "ep-20260701063020-vc7kb",
]


class EditRequest(BaseModel):
    image: str
    new_id: Optional[str] = None
    new_name: Optional[str] = None


async def call_ai(client: httpx.AsyncClient, image: str, prompt: str):
    """Call SeedEdit 3.0 via 火山方舟, try multiple model names, return result image URL."""
    last_error = ""
    for model_name in MODEL_CANDIDATES:
        logger.info(f"调用模型: {model_name}, prompt长度: {len(prompt)}, image大小: {len(image)//1024}KB")
        resp = await client.post(
            VOLC_URL,
            headers={"Authorization": f"Bearer {VOLC_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": model_name,
                "prompt": prompt,
                "image": image,
                "response_format": "url",
                "size": "2K",
                "watermark": False,
            },
            timeout=120,
        )
        logger.info(f"模型 {model_name} 返回状态: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json().get("data", [])
            url = (data[0].get("url") or data[0]) if data else None
            if url:
                return url
        d = resp.json() if resp.text else {}
        err_msg = d.get('error',{}).get('message','') or str(d)[:300]
        last_error = f"{resp.status_code}: {err_msg}"
        logger.warning(f"模型 {model_name} 失败: {last_error}")
    raise Exception(f"所有模型名均失败。最后一个错误: {last_error}")


async def url_to_base64(client: httpx.AsyncClient, url: str) -> str:
    r = await client.get(url)
    r.raise_for_status()
    b64 = base64.b64encode(r.content).decode()
    return f"data:image/png;base64,{b64}"


@app.post("/api/edit")
async def edit_image(req: EditRequest):
    if not req.new_id and not req.new_name:
        raise HTTPException(400, "至少填写姓名或身份证号中的一项")
    if not VOLC_API_KEY:
        raise HTTPException(500, "API Key 未配置")

    async with httpx.AsyncClient(timeout=180) as c:
        edits = []
        if req.new_name:
            edits.append(f"把「姓名」后面的名字改为「{req.new_name}」")

        birth = ""
        if req.new_id:
            edits.append(f"把「公民身份号码」后面的数字改为{req.new_id}")
            y, m, d = req.new_id[6:10], req.new_id[10:12], req.new_id[12:14]
            birth = f"{y}年{int(m)}月{int(d)}日"
            edits.append(f"把「出生」后面的日期改为{int(m)}月{int(d)}日")

        prompt = (
            f"请对证件照片进行以下修改：" + "；".join(edits)
            + "。要求：标签文字不要动；新文字颜色用灰绿色#959A8D；底纹、图案、边框等其他所有内容保持不变；输出清晰自然的照片。"
        )
        result_url = await call_ai(c, req.image, prompt)
        if not result_url:
            raise HTTPException(500, "编辑失败")
        result = await url_to_base64(c, result_url)

    return {"code": 0, "data": {
        "image": result,
        "birth_date": birth,
        "new_id": req.new_id,
        "new_name": req.new_name,
    }}


@app.get("/api/health")
def health():
    return {"status": "ok", "models": MODEL_CANDIDATES, "api_configured": bool(VOLC_API_KEY)}


app.mount("/", StaticFiles(directory="static", html=True), name="static")
