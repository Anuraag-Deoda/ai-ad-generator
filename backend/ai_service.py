import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_ad_script(product):
    prompt = f"""
You are an expert product copywriter. Write an engaging video script in 4 parts for the following product.

Product title: {product.get('title')}
Description: {product.get('description')}
Price: {product.get('price')}
Key features:
- {chr(10).join(product.get('features', []))}

Respond in this JSON format:
{{
  "hook": "Exciting hook (2-3 seconds)",
  "pitch": "Main product pitch (8-10 seconds)",
  "features": "Highlight top features (8-10 seconds)",
  "cta": "Call to action (3-5 seconds)"
}}
"""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You write catchy, persuasive product ad scripts for videos."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )

    content = response.choices[0].message['content']
    return eval(content)
