# test_api.py
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    base_url="https://api.meta.ai/v1",
    api_key=os.environ["MODEL_API_KEY"],
)

resp = client.chat.completions.create(
    model="muse-spark-1.1",  # swap for whatever your account lists
    messages=[{"role": "user", "content": "Say hello in Hindi."}],
)
print(resp.choices[0].message.content)
