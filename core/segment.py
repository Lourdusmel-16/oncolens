import os
import base64
import json
import re
import cv2
import requests
from dotenv import load_dotenv
from typing import Dict, Any, List

# Load .env automatically whenever this module is used
load_dotenv()
class LesionSegmenter:
    """
    Production wrapper for Meta SAM 3.1 over the Responses API.
    Streams token grammar, extracts bounding boxes, draws visual overlays,
    and calculates relative lesion area ratio.
    """
    def __init__(self, api_key: str = None, base_url: str = "https://api.meta.ai/v1"):
        self.api_key = api_key or os.getenv("MODEL_API_KEY")
        if not self.api_key:
            raise ValueError("MODEL_API_KEY is not set.")
        self.base_url = base_url

    def segment(self, image_path: str, primary_concept: str = "white patch") -> Dict[str, Any]:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at '{image_path}'")

        with open(image_path, "rb") as f:
            b64_image = base64.b64encode(f.read()).decode("utf-8")

        img = cv2.imread(image_path)
        img_h, img_w = img.shape[:2]
        total_pixels = img_h * img_w

        url = f"{self.base_url}/responses"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

        # Attempt primary concept, fallback to 'oral lesion' if no match
        concepts_to_try = [primary_concept, "oral lesion"] if primary_concept != "oral lesion" else ["oral lesion"]
        box_matches = []
        matched_concept = None

        for concept in concepts_to_try:
            payload = {
                "model": "sam-3.1",
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": concept},
                            {"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64_image}"},
                        ],
                    }
                ],
                "stream": True,
            }

            resp = requests.post(url, headers=headers, json=payload, stream=True)
            if resp.status_code != 200:
                continue

            raw_tokens = []
            for raw_line in resp.iter_lines(decode_unicode=False):
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8", errors="replace")
                if not line.startswith("data: "):
                    continue
                data = line.removeprefix("data: ")
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                    if event.get("type") == "response.output_text.delta":
                        raw_tokens.append(event.get("delta", ""))
                except json.JSONDecodeError:
                    continue

            stream_text = "".join(raw_tokens)
            matches = re.findall(r"<\|box;x1=(\d+);y1=(\d+);x2=(\d+);y2=(\d+);w=(\d+);h=(\d+)\|>", stream_text)
            if matches:
                box_matches = matches
                matched_concept = concept
                break

        overlay = img.copy()
        regions: List[Dict[str, int]] = []
        total_lesion_pixels = 0

        for i, (x1, y1, x2, y2, w, h) in enumerate(box_matches):
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            box_area = max(0, (x2 - x1)) * max(0, (y2 - y1))
            total_lesion_pixels += box_area

            regions.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "area_px": box_area})

            # Draw visual highlight for clinician reviewer view
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                overlay,
                f"SAM 3.1: {matched_concept}",
                (x1, max(y1 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        area_ratio = round(float(total_lesion_pixels / total_pixels), 4)

        return {
            "has_lesion_detected": len(regions) > 0,
            "matched_concept": matched_concept,
            "region_count": len(regions),
            "regions": regions,
            "area_ratio": area_ratio,
            "overlay_image": overlay,
        }