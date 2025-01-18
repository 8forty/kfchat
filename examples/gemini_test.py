import os

import dotenv
import openai
from openai import OpenAI

dotenv.load_dotenv(override=True)

client = OpenAI(
    api_key=os.getenv('GEMINI_API_KEY'),
    base_url=os.getenv('GEMINI_OPENAI_ENDPOINT'),  # "https://generativelanguage.googleapis.com/v1beta/openai/"
)

model_name = ['gemini-1.5-flash-002', 'gemini-1.5-flash', 'gemini-1.5-flash-8b', ][2]
print(f'model: {model_name}')
tries = 1
while True:
    try:
        response = client.chat.completions.create(
            model=model_name,
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
        print(f'response in {tries} tries: {response.choices[0].message}')
        # client.list_models()
        break
    except openai.RateLimitError as e:
        print('...')
        tries += 1
