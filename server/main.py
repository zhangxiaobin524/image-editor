import os
import io
import base64
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

FONT_PATH = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"


class EditRequest(BaseModel):
    image: str
    new_id: str


def parse_birth(id_num: str) -> str:
    y, m, d = id_num[6:10], id_num[10:12], id_num[12:14]
    return f"{y}年{int(m)}月{int(d)}日"


def b64_to_pil(b64: str) -> Image.Image:
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


def sample_bg_color(img: np.ndarray, x1, y1, x2, y2) -> tuple[int, int, int]:
    """Sample median color from edge pixels of region (likely background)."""
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return (220, 220, 220)
    # Take top and bottom 2-pixel strips as background
    roi = img[y1:y2, x1:x2]
    return tuple(int(c) for c in np.median(roi[:3, :, :], axis=(0, 1)))


def sample_text_color(img: np.ndarray, x1, y1, x2, y2) -> tuple[int, int, int]:
    """Sample dark text color from region."""
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return (80, 40, 40)
    roi = img[y1:y2, x1:x2]
    return tuple(int(c) for c in np.mean(roi[roi.mean(axis=2) < 120], axis=0))


def erase_and_draw(pil: Image.Image, draw: ImageDraw.ImageDraw,
                   x1: int, y1: int, x2: int, y2: int,
                   text: str, font: ImageFont.FreeTypeFont,
                   text_color: tuple[int, int, int],
                   bg_color: tuple[int, int, int]) -> None:
    """Cover old text with bg rectangle, then draw new text."""
    arr = np.array(pil)

    # Expand region slightly for clean coverage
    margin = 4
    ex1, ey1 = max(0, x1 - margin), max(0, y1 - margin)
    ex2, ey2 = min(pil.width, x2 + margin), min(pil.height, y2 + margin)

    # Paint background rectangle
    draw.rectangle([ex1, ey1, ex2, ey2], fill=bg_color)

    # Draw new text centered in region
    region_w, region_h = x2 - x1, y2 - y1
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    # Auto-scale font if too wide
    current_font = font
    if tw > region_w * 0.95:
        scale = region_w * 0.95 / tw
        new_size = max(10, int(font.size * scale))
        current_font = ImageFont.truetype(FONT_PATH, new_size)
        bbox = draw.textbbox((0, 0), text, font=current_font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    tx = int(x1 + (region_w - tw) / 2)
    ty = int(y1 + (region_h - th) / 2 - bbox[1])

    # For ID number: use monospace-like spacing to match original
    if len(text) == 18 and all(c.isdigit() or c.upper() == 'X' for c in text):
        char_w = region_w / len(text)
        for i, ch in enumerate(text):
            cx = int(x1 + i * char_w)
            draw.text((cx, ty), ch, font=current_font, fill=text_color)
    else:
        draw.text((tx, ty), text, font=current_font, fill=text_color)


@app.post("/api/edit")
async def edit_image(req: EditRequest):
    birth = parse_birth(req.new_id)

    if not os.path.exists(FONT_PATH):
        raise HTTPException(500, f"Font not found: {FONT_PATH}")

    pil = b64_to_pil(req.image)
    w, h = pil.size
    arr = np.array(pil)
    draw = ImageDraw.Draw(pil)

    # ---- Approximate positions by card proportions ----
    # These are rough estimates for a standard ID card
    # Birth date area: ~28%-58% width, ~52%-59% height
    birth_x1, birth_y1 = int(w * 0.28), int(h * 0.52)
    birth_x2, birth_y2 = int(w * 0.58), int(h * 0.59)

    # ID number area: ~12%-88% width, ~81%-90% height
    id_x1, id_y1 = int(w * 0.12), int(h * 0.81)
    id_x2, id_y2 = int(w * 0.88), int(h * 0.90)

    # Sample background colors
    birth_bg = sample_bg_color(arr, birth_x1, birth_y1, birth_x2, birth_y2)
    id_bg = sample_bg_color(arr, id_x1, id_y1, id_x2, id_y2)

    # Sample text colors
    birth_color = sample_text_color(arr, birth_x1, birth_y1, birth_x2, birth_y2)
    id_color = sample_text_color(arr, id_x1, id_y1, id_x2, id_y2)

    # Determine font sizes from region height
    birth_font_size = int((birth_y2 - birth_y1) * 0.65)
    id_font_size = int((id_y2 - id_y1) * 0.55)

    birth_font = ImageFont.truetype(FONT_PATH, birth_font_size)
    id_font = ImageFont.truetype(FONT_PATH, id_font_size)

    # Erase + redraw both fields
    erase_and_draw(pil, draw, birth_x1, birth_y1, birth_x2, birth_y2,
                   birth, birth_font, birth_color, birth_bg)
    erase_and_draw(pil, draw, id_x1, id_y1, id_x2, id_y2,
                   req.new_id, id_font, id_color, id_bg)

    return {
        "code": 0,
        "data": {
            "image": pil_to_b64(pil),
            "birth_date": birth,
            "new_id": req.new_id,
            "method": "pillow-render",
        }
    }


@app.get("/api/health")
def health():
    return {"status": "ok", "font": os.path.exists(FONT_PATH), "method": "pillow-render"}


app.mount("/", StaticFiles(directory="static", html=True), name="static")
