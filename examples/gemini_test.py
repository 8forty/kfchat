import os

import dotenv
from openai import OpenAI

dotenv.load_dotenv(override=True)

client = OpenAI(
    api_key=os.getenv('GEMINI_API_KEY'),
    base_url=os.getenv('GEMINI_OPENAI_ENDPOINT'),  # "https://generativelanguage.googleapis.com/v1beta/openai/"
)

response = client.chat.completions.create(
    model="gemini-1.5-flash-8b",  # gemini-1.5-flash",
    n=1,
    max_tokens=100,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "Explain to me how AI works"
        }
    ]
)

print(response.choices[0].message)
