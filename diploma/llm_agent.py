import os
import re
import time
import backoff
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APITimeoutError, APIError

load_dotenv()

ARXIV_API_RATE_LIMIT_SEC = 3.0 
MAX_NUM_TOKENS = 4096
OPENROUTER_API_BASE_URL = os.getenv("OPENROUTER_API_BASE_URL")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL")

if not OPENROUTER_API_KEY:
    raise EnvironmentError("OPENROUTER_API_KEY не установлен. Поместите его в .env или переменные окружения.")
if not OPENROUTER_API_BASE_URL:
    raise EnvironmentError("OPENROUTER_API_BASE_URL не установлен. Поместите его в .env или переменные окружения.")
if not OPENROUTER_MODEL:
    raise EnvironmentError("OPENROUTER_MODEL не установлен. Поместите его в .env или переменные окружения.")


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


@backoff.on_exception(backoff.expo, (RateLimitError, APITimeoutError, APIError), max_time=45)
def get_response_from_llm(
        msg,
        print_debug=False,
        msg_history=None,
        temperature=0.7,
):
    if msg_history is None:
        msg_history = []

    client = OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_API_BASE_URL,
    )

    new_msg_history = msg_history + [{"role": "user", "content": msg}]

    try:
        response = client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
        )

        content = response.choices[0].message.content
        if content:
            content = clean_text(content)
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    
    except RateLimitError as e:
        print(f"[ERROR] Rate limit exceeded (429): {e}")
        raise
    except APITimeoutError as e:
        print(f"[ERROR] API timeout: {e}")
        raise
    except APIError as e:
        print(f"[ERROR] API error: {e}")
        if "401" in str(e) or "Unauthorized" in str(e):
            print("Проверьте OPENROUTER_API_KEY в .env")
        elif "404" in str(e):
            print(f"Модель '{OPENROUTER_MODEL}' не найдена")
        raise

    if print_debug:
        print()
        print("*" * 20 + " LLM START " + "*" * 20)
        for j, m in enumerate(new_msg_history):
            preview = m["content"][:80].replace('\n', ' ')
            print(f'{j}, {m["role"]}: {preview}...')
        print(content)
        print("*" * 21 + " LLM END " + "*" * 21)
        print()
    
    time.sleep(ARXIV_API_RATE_LIMIT_SEC)
    
    return content, new_msg_history 
