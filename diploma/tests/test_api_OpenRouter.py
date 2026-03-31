import os
import requests
import json
import datetime
import re
from dotenv import load_dotenv

load_dotenv()


def get_response_from_llm(role, prompt):
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        raise EnvironmentError("OPENROUTER_API_KEY is not set. Put it in .env or environment variables.")

    # OpenRouter LLM provider 
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {openrouter_api_key}",
            "Content-Type": "application/json"
        },
        data=json.dumps({
            "model": "qwen/qwen3.6-plus-preview:free",
            "messages": [
            {
                "role": role,
                "content": prompt
            }
            ],
            "reasoning": {"enabled": True}
        })
    )
    
    return response.json()['choices'][0]['message']['content']


def clean_text(text):
    # Убираем ** с обеих сторон
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    
    # Убираем * для курсива
    text = re.sub(r'\*(.*?)\*', r'\1', text)

    # Убираем `` для кода
    text = re.sub(r'\`(.*?)\`', r'\1', text)

    # Убираем ### заголовки
    text = re.sub(r'^###\s*', '', text, flags=re.MULTILINE)

    # Убираем * в начале списков
    text = re.sub(r'^\*\s+', '', text, flags=re.MULTILINE)

    # Убираем * в конце строк (если это форматирование, но не знак умножения внутри текста)
    text = re.sub(r'\*\s*$', '', text, flags=re.MULTILINE)

    return text

role = 'user'
prompt = 'Какие научные статьи реальных авторов можно прочитать, чтобы получить представление в области нейронных сетей типа Трансформеры?'

start = datetime.datetime.now()
response = get_response_from_llm(role, prompt)
finish = datetime.datetime.now()
print('Время работы: ' + str(finish - start))

print(clean_text(response))