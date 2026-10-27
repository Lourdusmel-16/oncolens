import os
import base64
import json
from dotenv import load_dotenv
from openai import OpenAI

# 1. Load environment variables
load_dotenv()
api_key = os.getenv("MODEL_API_KEY")

if not api_key:
    raise ValueError("MODEL_API_KEY is missing from your .env file!")

# 2. Configure OpenAI client for Meta Model API
client = OpenAI(
    base_url="https://api.meta.ai/v1",
    api_key=api_key,
)

image_path = "data/test_images/sample1.jpeg"

if not os.path.exists(image_path):
    raise FileNotFoundError(f"Missing test image at '{image_path}'. Place an image there first.")

# 3. Base64 encode the image
with open(image_path, "rb") as f:
    b64_image = base64.b64encode(f.read()).decode("utf-8")

print(f"[*] Loaded image: {image_path} ({len(b64_image)} b64 chars)")
print("[*] Sending multimodal request to Meta Model API...")

# 4. Multimodal call
# Note: Verify your exact model name from your dev.meta.ai dashboard
# (e.g., 'meta-llama/Llama-3.2-11B-Vision-Instruct' or the hackathon-provided model ID)
MODEL_NAME = "muse-spark-1.3" 

system_prompt = (
    "You are a clinical lesion documentation copilot for frontline health workers. "
    "You do NOT diagnose pathology. You objectively describe visual features (color, margins, texture, anatomical site) "
    "and output strictly valid JSON matching this schema:\n"
    "{\n"
    '  "visible_findings": ["feature 1", "feature 2"],\n'
    '  "image_quality": "good | acceptable | poor",\n'
    '  "quality_issues": ["glare", "blur"] or [],\n'
    '  "clinical_note": "objective description"\n'
    "}"
)

try:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text": "Describe the visible lesion characteristics in this intraoral photo. Do not diagnose."
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}
                    },
                ],
            },
        ],
        temperature=0.1,  # Keep low for reproducible, factual outputs
    )

    raw_content = response.choices[0].message.content.strip()
    print("\n--- Raw Response ---")
    print(raw_content)

    # 5. Clean Markdown fences if present
    cleaned = raw_content
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1]
    if cleaned.endswith("```"):
        cleaned = cleaned.rsplit("\n", 1)[0]
    cleaned = cleaned.strip()

    # 6. Parse and validate JSON
    parsed = json.loads(cleaned)
    print("\n--- Parsed JSON Output ---")
    print(json.dumps(parsed, indent=2))
    print("\n[SUCCESS] Day 1 Multimodal Smoke Test Passed!")

except json.JSONDecodeError:
    print("\n[ERROR] Model output was not valid JSON:")
    print(raw_content)
except Exception as e:
    print(f"\n[ERROR] API call failed: {e}")