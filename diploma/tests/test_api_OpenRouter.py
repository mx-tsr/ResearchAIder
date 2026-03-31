import requests
import json
import datetime
import re


def get_response_from_llm(role, prompt):
    # OpenRouter LLM provider 
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            # My personal API - should move to .env
            "Authorization": "",
            "Content-Type": "application/json"
        },
        data=json.dumps({
            "model": "deepseek/deepseek-r1-0528:free",
            "messages": [
            {
                "role": role,
                "content": prompt
            }
            ]
        })
    )
    
    return response


def clean_text(text):
    # Убираем ** с обеих сторон
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    
    # Убираем * для курсива
    text = re.sub(r'\*(.*?)\*', r'\1', text)

    # Убираем `` для кода
    text = re.sub(r'\`(.*?)\`', r'\1', text)

    return text

role = "user"
prompt = "Какие научные статьи реальных авторов можно прочитать, чтобы получить представление в области нейронных сетей типа Трансформеры."

start = datetime.datetime.now()
response = get_response_from_llm(role, prompt)
finish = datetime.datetime.now()

print('Время работы: ' + str(finish - start))

data = json.loads(response.content)
content = data["choices"][0]["message"]["content"]
print(clean_text(content))