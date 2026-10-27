import os
import base64
import json
import re
import cv2
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("MODEL_API_KEY")
if not api_key:
    raise ValueError("MODEL_API_KEY is missing from .env!")

# Support either .jpeg or .jpg
image_path = "data/test_images/sample1.jpeg"
if not os.path.exists(image_path):
    image_path = "data/test_images/sample1.jpg"

print(f"[*] Reading image: {image_path}")
with open(image_path, "rb") as f:
    b64_image = base64.b64encode(f.read()).decode("utf-8")

# Specific concept for buccal mucosa
CONCEPT = "white patch"
print(f"[*] Calling SAM 3.1 with concept: '{CONCEPT}'...")

url = "https://api.meta.ai/v1/responses"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
    "Accept": "text/event-stream",
}

payload = {
    "model": "sam-3.1",
    "input": [
        {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": CONCEPT},
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{b64_image}",
                },
            ],
        }
    ],
    "stream": True,
}

response = requests.post(url, headers=headers, json=payload, stream=True)
response.raise_for_status()

raw_tokens = []
print("[*] Streaming response from Meta Model API...")

for raw_line in response.iter_lines(decode_unicode=False):
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

full_stream = "".join(raw_tokens)

# Extract bounding box: <|box;x1=...;y1=...;x2=...;y2=...;w=...;h=...|>
box_matches = re.findall(r"<\|box;x1=(\d+);y1=(\d+);x2=(\d+);y2=(\d+);w=(\d+);h=(\d+)\|>", full_stream)

if not box_matches:
    print(f"\n[!] No match for '{CONCEPT}'. Retrying with concept 'oral lesion'...")
    # Fallback to general concept if white patch doesn't match
    payload["input"][0]["content"][0]["text"] = "oral lesion"
    resp2 = requests.post(url, headers=headers, json=payload, stream=True)
    tokens2 = []
    for raw_line in resp2.iter_lines(decode_unicode=False):
        if not raw_line: continue
        line = raw_line.decode("utf-8", errors="replace")
        if not line.startswith("data: "): continue
        data = line.removeprefix("data: ")
        if data == "[DONE]": break
        try:
            event = json.loads(data)
            if event.get("type") == "response.output_text.delta":
                tokens2.append(event.get("delta", ""))
        except: continue
    full_stream = "".join(tokens2)
    box_matches = re.findall(r"<\|box;x1=(\d+);y1=(\d+);x2=(\d+);y2=(\d+);w=(\d+);h=(\d+)\|>", full_stream)

if not box_matches:
    print("[!] Still no box detected. Make sure the photo shows clear lesion tissue.")
    exit()

print(f"\n[+] SUCCESS! SAM 3.1 identified {len(box_matches)} lesion region(s).")

# Load image and draw the box
img = cv2.imread(image_path)
overlay = img.copy()

for i, (x1, y1, x2, y2, w, h) in enumerate(box_matches):
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    print(f" -> Region #{i+1}: Top-Left=({x1}, {y1}), Bottom-Right=({x2}, {y2})")
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 3)
    cv2.putText(
        overlay,
        f"Lesion #{i+1}",
        (x1, max(y1 - 10, 25)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

os.makedirs("outputs", exist_ok=True)
output_path = "outputs/sample1_detected.jpg"
cv2.imwrite(output_path, overlay)

print(f"\n[DONE] Overwritten {output_path} with the new buccal mucosa lesion box!")
