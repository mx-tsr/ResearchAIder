import os
import re
import time
import backoff
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APITimeoutError, APIError

load_dotenv()

OPENROUTER_API_RATE_LIMIT_SEC = 3.0 
MAX_NUM_TOKENS = 4096
OPENROUTER_API_BASE_URL = os.getenv("OPENROUTER_API_BASE_URL")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL")
OPENROUTER_API_BASE_URL_FALLBACK = os.getenv("OPENROUTER_API_BASE_URL_FALLBACK")
OPENROUTER_API_KEY_FALLBACK = os.getenv("OPENROUTER_API_KEY_FALLBACK")

# Список API конфигураций: [(key, url), ...]
OPENROUTER_APIS = [(OPENROUTER_API_KEY, OPENROUTER_API_BASE_URL)]
if OPENROUTER_API_KEY_FALLBACK and OPENROUTER_API_BASE_URL_FALLBACK:
    OPENROUTER_APIS.append((OPENROUTER_API_KEY_FALLBACK, OPENROUTER_API_BASE_URL_FALLBACK))

if not OPENROUTER_API_KEY:
    raise EnvironmentError("OPENROUTER_API_KEY не установлен. Поместите его в .env или переменные окружения.")
if not OPENROUTER_API_BASE_URL:
    raise EnvironmentError("OPENROUTER_API_BASE_URL не установлен. Поместите его в .env или переменные окружения.")
if not OPENROUTER_MODEL:
    raise EnvironmentError("OPENROUTER_MODEL не установлен. Поместите его в .env или переменные окружения.")


def backoff_handler(details):
    """Логирует попытки backoff при rate limit"""
    print("[DEBUG] Попытка #{tries}, ждём {wait:0.1f} сек...".format(**details))


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


def call_llm_model(api_key, api_url, model, messages, temperature):
    client = OpenAI(
        api_key=api_key,
        base_url=api_url,
    )
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=MAX_NUM_TOKENS,
    )
    return response.choices[0].message.content


@backoff.on_exception(backoff.expo, (RateLimitError, APITimeoutError, APIError), max_time=90, on_backoff=backoff_handler)
def get_response_from_llm(
        msg,
        print_debug=False,
        msg_history=None,
        temperature=0.7,
        rate_limit=OPENROUTER_API_RATE_LIMIT_SEC
):
    if msg_history is None:
        msg_history = []

    new_msg_history = msg_history + [{"role": "user", "content": msg}]
    last_error = None

    for api_key, api_url in OPENROUTER_APIS:
        try:
            content = call_llm_model(
                api_key, 
                api_url, 
                OPENROUTER_MODEL, 
                new_msg_history, 
                temperature
            )

            if content:
                content = clean_text(content)
            new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]

            print(f'[DEBUG] получен ответ от LLM (api={api_url}, model={OPENROUTER_MODEL}): {content}')
            
            if print_debug:
                print()
                print("*" * 20 + " LLM START " + "*" * 20)
                for j, m in enumerate(new_msg_history):
                    preview = m["content"].replace('\n', ' ')
                    print(f'{j}, {m["role"]}: {preview}...')
                print(content)
                print("*" * 21 + " LLM END " + "*" * 21)
                print()

            time.sleep(rate_limit)

            return content, new_msg_history
        
        except RateLimitError as e:
            print(f"[ERROR] Rate limit 429 для API {api_url}: {e}")
            last_error = e

            if "Rate limit exceeded: free-models-per-day. Add 10 credits to unlock 1000 free model requests per day" in str(e) and OPENROUTER_API_KEY_FALLBACK and OPENROUTER_API_BASE_URL_FALLBACK:
                print("[DEBUG] Переключаюсь на fallback-API после исчерпания запросов основного API")
                content = call_llm_model(
                    OPENROUTER_API_KEY_FALLBACK,
                    OPENROUTER_API_BASE_URL_FALLBACK,
                    OPENROUTER_MODEL,
                    new_msg_history,
                    temperature
                )
            else:
                raise

            if (api_key, api_url) != OPENROUTER_APIS[-1]:
                print(f"[DEBUG] Переключаюсь на fallback-API")
                continue
            raise

        except APITimeoutError as e:
            print(f"[ERROR] API timeout для API {api_url}: {e}")
            last_error = e
            raise

        except APIError as e:
            print(f"[ERROR] API error для API {api_url}: {e}")
            if "401" in str(e) or "Unauthorized" in str(e):
                print("Проверьте OPENROUTER_API_KEY в .env")
            elif "404" in str(e):
                print(f"Модель '{OPENROUTER_MODEL}' не найдена")
            last_error = e
            raise

    if last_error:
        raise last_error

    raise RuntimeError("[ERROR] ни один LLM API не был вызван")
