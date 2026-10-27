import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    base_url="https://api.meta.ai/v1",
    api_key=os.environ["MODEL_API_KEY"],
)

print("[*] Fetching available models for your API key...")
models = client.models.list()
for m in models.data:
    print(f" -> {m.id}")