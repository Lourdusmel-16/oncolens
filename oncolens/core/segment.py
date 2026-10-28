import os
import re
import json
import time
import base64
import hashlib
from typing import Dict, Any, List, Optional, Sequence

import cv2
import numpy as np
import requests
from dotenv import load_dotenv
from meta_sam_parser import SegmentationMask, decode_mask_to_raster

from core.utils import load_resized

load_dotenv()

# box + mask token pair, as emitted by sam-3.1 (c= is present when confidence is requested)
RECORD_RE = re.compile(
    r"<\|box;x1=(\d+);y1=(\d+);x2=(\d+);y2=(\d+);w=(\d+);h=(\d+)(?:;c=([\d.]+))?\|>"
    r"<\|mask;x=\d+;y=\d+;(?:c=[\d.]+;)?data=(\d+),(\d+),([^|<>;,\"\\]+)\|>"
)

DEFAULT_CONCEPTS = ("white patch", "red patch", "ulcer")


class LesionSegmenter:
    """SAM 3.1 wrapper: one request per concept -> real pixel masks (not just boxes)."""

    def __init__(self, api_key: Optional[str] = None,
                 base_url: str = "https://api.meta.ai/v1",
                 cache_dir: str = "outputs/cache"):
        self.api_key = api_key or os.getenv("MODEL_API_KEY")
        if not self.api_key:
            raise ValueError("MODEL_API_KEY is not set.")
        self.base_url = base_url
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    # ---------- network ----------
    def _call_sam(self, jpeg_bytes: bytes, concept: str, use_cache: bool = True) -> str:
        key = hashlib.sha256(jpeg_bytes + concept.encode()).hexdigest()[:24]
        cache_path = os.path.join(self.cache_dir, f"sam_{key}.txt")
        if use_cache and os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                return f.read()

        b64 = base64.b64encode(jpeg_bytes).decode("utf-8")
        payload = {
            "model": "sam-3.1",
            "input": [{"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": concept},
                {"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"},
            ]}],
            "stream": True,
            "metadata": {"include_confidence": "true", "mask_encoding": "one_bit"},
        }
        headers = {"Authorization": f"Bearer {self.api_key}",
                   "Content-Type": "application/json",
                   "Accept": "text/event-stream"}

        last_err = None
        for attempt in range(3):
            try:
                resp = requests.post(f"{self.base_url}/responses", headers=headers,
                                     json=payload, stream=True, timeout=120)
                if resp.status_code == 200:
                    break
                last_err = f"HTTP {resp.status_code}: {resp.text[:300]}"
                if resp.status_code not in (429, 500, 502, 503, 504):
                    raise RuntimeError(f"SAM request failed ({last_err})")
            except requests.RequestException as e:
                last_err = str(e)
            time.sleep(2 * (attempt + 1))
        else:
            raise RuntimeError(f"SAM request failed after retries ({last_err})")

        parts: List[str] = []
        for raw in resp.iter_lines(decode_unicode=False):
            if not raw:
                continue
            line = raw.decode("utf-8", errors="replace")
            if not line.startswith("data: "):
                continue
            data = line[len("data: "):]
            if data == "[DONE]":
                break
            try:
                ev = json.loads(data)
            except json.JSONDecodeError:
                continue
            if ev.get("type") == "response.output_text.delta":
                parts.append(ev.get("delta", ""))
        text = "".join(parts)
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(text)
        return text

    # ---------- parsing ----------
    @staticmethod
    def parse_regions(stream_text: str, img_w: int, img_h: int, concept: str) -> List[Dict[str, Any]]:
        regions = []
        for m in RECORD_RE.finditer(stream_text):
            x1, y1, x2, y2, w, h, conf, mh, mw, payload = m.groups()
            x1, y1, x2, y2, w, h, mh, mw = map(int, (x1, y1, x2, y2, w, h, mh, mw))
            enc = "lossless" if payload[0] == "~" else "one_bit"
            small = np.frombuffer(
                decode_mask_to_raster(SegmentationMask(encoding=enc, payload=payload,
                                                       width=mw, height=mh)),
                dtype=np.uint8).reshape(mh, mw)
            # The response reports the frame size it saw; rescale if it differs from ours.
            sx, sy = img_w / w, img_h / h
            full = np.zeros((img_h, img_w), dtype=np.uint8)
            if abs(sx - 1) < 1e-6 and abs(sy - 1) < 1e-6:
                bx1, by1 = x1, y1
            else:
                bx1, by1 = int(x1 * sx), int(y1 * sy)
                small = cv2.resize(small, (max(1, int(mw * sx)), max(1, int(mh * sy))),
                                   interpolation=cv2.INTER_NEAREST)
            hh, ww = small.shape
            ye, xe = min(img_h, by1 + hh), min(img_w, bx1 + ww)
            full[by1:ye, bx1:xe] = small[: ye - by1, : xe - bx1]
            regions.append({
                "concept": concept,
                "box": [int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)],
                "confidence": float(conf) if conf is not None else None,
                "mask": full,
                "area_px": int(full.sum()),
            })
        return regions

    # ---------- public ----------
    def segment(self, image_path: str,
                concepts: Sequence[str] = DEFAULT_CONCEPTS,
                min_confidence: float = 0.3,
                use_cache: bool = True) -> Dict[str, Any]:
        bgr, jpeg = load_resized(image_path)
        img_h, img_w = bgr.shape[:2]

        regions: List[Dict[str, Any]] = []
        for concept in concepts:
            text = self._call_sam(jpeg, concept, use_cache=use_cache)
            for r in self.parse_regions(text, img_w, img_h, concept):
                if r["confidence"] is None or r["confidence"] >= min_confidence:
                    regions.append(r)

        for i, r in enumerate(regions):
            r["id"] = i
        return {"image_bgr": bgr, "regions": regions, "img_w": img_w, "img_h": img_h}

    # ---------- reviewer helpers (used by the app) ----------
    @staticmethod
    def area_ratio(regions: List[Dict[str, Any]], selected_ids: Optional[Sequence[int]] = None) -> float:
        """True pixel area of the UNION of selected masks (no double counting)."""
        chosen = [r for r in regions if selected_ids is None or r["id"] in selected_ids]
        if not chosen:
            return 0.0
        union = np.zeros_like(chosen[0]["mask"], dtype=np.uint8)
        for r in chosen:
            union |= r["mask"]
        return round(float(union.sum()) / union.size, 4)

    @staticmethod
    def render_overlay(bgr: np.ndarray, regions: List[Dict[str, Any]],
                       selected_ids: Optional[Sequence[int]] = None) -> np.ndarray:
        out = bgr.copy()
        palette = {"white patch": (0, 215, 255), "red patch": (0, 0, 255), "ulcer": (255, 0, 255)}
        for r in regions:
            if selected_ids is not None and r["id"] not in selected_ids:
                continue
            color = palette.get(r["concept"], (0, 255, 0))
            m = r["mask"].astype(bool)
            tint = out.copy()
            tint[m] = color
            out = cv2.addWeighted(tint, 0.35, out, 0.65, 0)
            cnts, _ = cv2.findContours(r["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, cnts, -1, color, 2)
            x1, y1, _, _ = r["box"]
            conf = f" {r['confidence']:.2f}" if r["confidence"] is not None else ""
            cv2.putText(out, f"#{r['id']} {r['concept']}{conf}", (x1, max(y1 - 6, 16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
        return out
