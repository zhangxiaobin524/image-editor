import os
import io
import base64
import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageDraw, ImageFont
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

FONT_PATH = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"


class EditRequest(BaseModel):
    image: str       # base64 ID card image
    new_id: str      # new 18-digit ID number


def parse_birth_from_id(id_num: str) -> str:
    y, m, d = id_num[6:10], id_num[10:12], id_num[12:14]
    return f"{y}年{int(m)}月{int(d)}日"


def b64_to_cv2(b64: str) -> np.ndarray:
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    buf = np.frombuffer(base64.b64decode(b64), np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def cv2_to_b64(img: np.ndarray) -> str:
    _, buf = cv2.imencode('.png', img)
    return f"data:image/png;base64,{base64.b64encode(buf).decode()}"


def ocr_lines(img: np.ndarray) -> list[dict]:
    """Run Tesseract, return list of detected text lines with bboxes."""
    data = pytesseract.image_to_data(
        img, lang='chi_sim', output_type=pytesseract.Output.DICT,
        config='--psm 6'
    )
    lines = {}
    for i in range(len(data['text'])):
        t = data['text'][i].strip()
        if not t or int(data['conf'][i]) < 20:
            continue
        key = (data['block_num'][i], data['line_num'][i])
        if key not in lines:
            lines[key] = {
                'left': data['left'][i], 'top': data['top'][i],
                'right': data['left'][i] + data['width'][i],
                'bottom': data['top'][i] + data['height'][i],
                'text': t, 'words': [t]
            }
        else:
            r = lines[key]
            r['left'] = min(r['left'], data['left'][i])
            r['top'] = min(r['top'], data['top'][i])
            r['right'] = max(r['right'], data['left'][i] + data['width'][i])
            r['bottom'] = max(r['bottom'], data['top'][i] + data['height'][i])
            r['text'] += t
            r['words'].append(t)
    return sorted(lines.values(), key=lambda x: x['top'])


def find_region(lines: list[dict], keyword: str) -> dict | None:
    """Find line containing keyword."""
    for r in lines:
        if keyword in r['text']:
            return r
    return None


def find_id_number_region(lines: list[dict]) -> dict | None:
    """Find line with 17+ digit/chars near bottom."""
    for r in reversed(lines):
        digits = ''.join(c for c in r['text'] if c.isdigit() or c.upper() == 'X')
        if len(digits) >= 17:
            return r
    return None


def sample_text_color(img: np.ndarray, region: dict) -> tuple[int, int, int]:
    """Sample dominant text color (BGR) from region."""
    x1, y1 = max(0, region['left']), max(0, region['top'])
    x2, y2 = min(img.shape[1], region['right']), min(img.shape[0], region['bottom'])
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return (80, 40, 40)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = bin_img == 0
    if np.sum(dark) > 20:
        return tuple(int(c) for c in np.mean(roi[dark], axis=0))
    return (80, 40, 40)


def erase_text(img: np.ndarray, region: dict, margin: int = 6) -> np.ndarray:
    """Inpaint text area using TELEA algorithm."""
    h, w = img.shape[:2]
    x1 = max(0, region['left'] - margin)
    y1 = max(0, region['top'] - margin)
    x2 = min(w, region['right'] + margin)
    y2 = min(h, region['bottom'] + margin)
    roi = img[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    bin_img = cv2.dilate(bin_img, kernel, iterations=3)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y1:y2, x1:x2] = bin_img
    return cv2.inpaint(img, mask, 7, cv2.INPAINT_TELEA)


def render_new_text(img: np.ndarray, region: dict, text: str, font_path: str,
                    color_bgr: tuple[int, int, int]) -> np.ndarray:
    """Render new text centered in region."""
    region_w = region['right'] - region['left']
    region_h = region['bottom'] - region['top']

    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)

    # Determine font size
    font_size = int(region_h * 0.65)
    font = ImageFont.truetype(font_path, font_size)
    test_bbox = draw.textbbox((0, 0), text, font=font)
    tw = test_bbox[2] - test_bbox[0]

    if tw > region_w * 1.15:
        scale = region_w * 1.15 / tw
        font = ImageFont.truetype(font_path, max(10, int(font_size * scale)))
        test_bbox = draw.textbbox((0, 0), text, font=font)
        tw = test_bbox[2] - test_bbox[0]

    th = test_bbox[3] - test_bbox[1]
    x = int(region['left'] + (region_w - tw) / 2)
    y = int(region['top'] + (region_h - th) / 2 - test_bbox[1])

    # Convert BGR -> RGB for PIL
    rgb = (color_bgr[2], color_bgr[1], color_bgr[0])

    if len(text) == 18 and all(c.isdigit() or c.upper() == 'X' for c in text):
        # Monospace rendering for ID number
        char_w = region_w / len(text)
        for i, ch in enumerate(text):
            cx = int(region['left'] + i * char_w)
            draw.text((cx, y), ch, font=font, fill=rgb)
    else:
        draw.text((x, y), text, font=font, fill=rgb)

    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


@app.post("/api/edit")
async def edit_image(req: EditRequest):
    birth = parse_birth_from_id(req.new_id)

    if not os.path.exists(FONT_PATH):
        raise HTTPException(500, f"Font not found: {FONT_PATH}")

    img = b64_to_cv2(req.image)
    h, w = img.shape[:2]

    # ---- OCR ----
    lines = ocr_lines(img)

    birth_region = find_region(lines, "出生") or find_region(lines, "生")
    id_region = find_id_number_region(lines)

    # Fallback: estimate by proportions
    if birth_region is None:
        birth_region = {
            'left': int(w * 0.28), 'top': int(h * 0.52),
            'right': int(w * 0.58), 'bottom': int(h * 0.59),
        }
    if id_region is None:
        id_region = {
            'left': int(w * 0.12), 'top': int(h * 0.81),
            'right': int(w * 0.88), 'bottom': int(h * 0.90),
        }

    birth_original = birth_region.copy()
    id_original = id_region.copy()

    # ---- Sample colors ----
    birth_color = sample_text_color(img, birth_original)
    id_color = sample_text_color(img, id_original)

    # ---- Erase old text ----
    img = erase_text(img, birth_original)
    img = erase_text(img, id_original)

    # ---- Render new text ----
    img = render_new_text(img, birth_original, birth, FONT_PATH, birth_color)
    img = render_new_text(img, id_original, req.new_id, FONT_PATH, id_color)

    return {
        "code": 0,
        "data": {
            "image": cv2_to_b64(img),
            "birth_date": birth,
            "method": "cv-pipeline",
            "ocr_used": birth_region is not None and id_region is not None,
        }
    }


@app.get("/api/health")
def health():
    import subprocess
    t_ok = False
    try:
        r = subprocess.run(['tesseract', '--version'], capture_output=True, text=True)
        t_ok = r.returncode == 0
    except Exception:
        pass
    return {
        "status": "ok",
        "tesseract": t_ok,
        "font_available": os.path.exists(FONT_PATH),
        "method": "cv-pipeline"
    }


app.mount("/", StaticFiles(directory="static", html=True), name="static")
