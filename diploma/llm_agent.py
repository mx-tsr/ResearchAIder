import os
import re
import time
import backoff
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APITimeoutError, APIError

load_dotenv()

OPENROUTER_API_RATE_LIMIT_SEC = 3.0 
MAX_NUM_TOKENS = 16384
OPENROUTER_API_BASE_URL = os.getenv("OPENROUTER_API_BASE_URL")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL")

OPENROUTER_API_BASE_URL_FALLBACK_1 = os.getenv("OPENROUTER_API_BASE_URL_FALLBACK_1")
OPENROUTER_API_KEY_FALLBACK_1 = os.getenv("OPENROUTER_API_KEY_FALLBACK_1")
OPENROUTER_MODEL_FALLBACK_1 = os.getenv("OPENROUTER_MODEL_FALLBACK_1")

OPENROUTER_API_BASE_URL_FALLBACK_2 = os.getenv("OPENROUTER_API_BASE_URL_FALLBACK_2")
OPENROUTER_API_KEY_FALLBACK_2 = os.getenv("OPENROUTER_API_KEY_FALLBACK_2")
OPENROUTER_MODEL_FALLBACK_2 = os.getenv("OPENROUTER_MODEL_FALLBACK_2")

OPENROUTER_API_BASE_URL_FALLBACK_3 = os.getenv("OPENROUTER_API_BASE_URL_FALLBACK_3")
OPENROUTER_API_KEY_FALLBACK_3 = os.getenv("OPENROUTER_API_KEY_FALLBACK_3")
OPENROUTER_MODEL_FALLBACK_3 = os.getenv("OPENROUTER_MODEL_FALLBACK_3")

OPENROUTER_API_BASE_URL_FALLBACK_4 = os.getenv("OPENROUTER_API_BASE_URL_FALLBACK_4")
OPENROUTER_API_KEY_FALLBACK_4 = os.getenv("OPENROUTER_API_KEY_FALLBACK_4")
OPENROUTER_MODEL_FALLBACK_4 = os.getenv("OPENROUTER_MODEL_FALLBACK_4")

MODELLGATE_API_BASE_URL = os.getenv("MODELLGATE_API_BASE_URL")
MODELLGATE_API_KEY = os.getenv("MODELLGATE_API_KEY")
MODELLGATE_MODEL = os.getenv("MODELLGATE_MODEL")

# Список API конфигураций: [(key, url), ...]
OPENROUTER_APIS = [(OPENROUTER_API_KEY, OPENROUTER_API_BASE_URL, OPENROUTER_MODEL)]
if OPENROUTER_API_KEY_FALLBACK_1 and OPENROUTER_API_BASE_URL_FALLBACK_1:
    OPENROUTER_APIS.append((OPENROUTER_API_KEY_FALLBACK_1, OPENROUTER_API_BASE_URL_FALLBACK_1, OPENROUTER_MODEL_FALLBACK_1))
if OPENROUTER_API_KEY_FALLBACK_2 and OPENROUTER_API_BASE_URL_FALLBACK_2:
    OPENROUTER_APIS.append((OPENROUTER_API_KEY_FALLBACK_2, OPENROUTER_API_BASE_URL_FALLBACK_2, OPENROUTER_MODEL_FALLBACK_2))
if OPENROUTER_API_KEY_FALLBACK_3 and OPENROUTER_API_BASE_URL_FALLBACK_3:
    OPENROUTER_APIS.append((OPENROUTER_API_KEY_FALLBACK_3, OPENROUTER_API_BASE_URL_FALLBACK_3, OPENROUTER_MODEL_FALLBACK_3))
if OPENROUTER_API_KEY_FALLBACK_4 and OPENROUTER_API_BASE_URL_FALLBACK_4:
    OPENROUTER_APIS.append((OPENROUTER_API_KEY_FALLBACK_4, OPENROUTER_API_BASE_URL_FALLBACK_4, OPENROUTER_MODEL_FALLBACK_4))
if MODELLGATE_API_KEY and MODELLGATE_API_BASE_URL:
    OPENROUTER_APIS.append((MODELLGATE_API_KEY, MODELLGATE_API_BASE_URL, MODELLGATE_MODEL))


if not OPENROUTER_API_KEY:
    raise EnvironmentError("OPENROUTER_API_KEY не установлен. Поместите его в .env или переменные окружения.")
if not OPENROUTER_API_BASE_URL:
    raise EnvironmentError("OPENROUTER_API_BASE_URL не установлен. Поместите его в .env или переменные окружения.")
if not OPENROUTER_MODEL:
    raise EnvironmentError("OPENROUTER_MODEL не установлен. Поместите его в .env или переменные окружения.")

API_DAILY_LIMIT_EXHAUSTED = [False] * len(OPENROUTER_APIS)


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

    for api_idx, (api_key, api_url, api_model) in enumerate(OPENROUTER_APIS):

        if API_DAILY_LIMIT_EXHAUSTED[api_idx]:
            continue

        try:
            content = call_llm_model(
                api_key, 
                api_url, 
                api_model, 
                new_msg_history, 
                temperature
            )

            if content:
                content = clean_text(content)
            new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]

            print(f'[DEBUG] получен ответ от LLM (idx={api_idx+1}, api={api_url}, model={api_model}):\n {"-"*50}\n {content}\n {"-"*50}\n')
            
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

            if "Rate limit exceeded: free-models-per-day. Add 10 credits to unlock 1000 free model requests per day" in str(e):
                API_DAILY_LIMIT_EXHAUSTED[api_idx] = True
                print(f"[DEBUG] API #{api_idx+1} исчерпал дневной лимит")
                continue
            else:
                # Другие RateLimitError обрабатываются через backoff
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
