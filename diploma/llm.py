import re
import time
import backoff
from json import JSONDecodeError
from openai import OpenAI, RateLimitError, APITimeoutError, APIError

from config import OPENROUTER_APIS, OPENROUTER_API_RATE_LIMIT_SEC
from utils import load_logger, record_token_usage


logger = load_logger()

API_DAILY_LIMIT_EXHAUSTED = [False] * len(OPENROUTER_APIS)


def backoff_handler(details):
    """
    Логирует попытки backoff при rate limit ошибках
    """
    logger.info(f"Попытка #{details['tries']}, ждём {details['wait']:.1f} сек...")


def clean_text(text):
    '''
    Очищает текст от Markdown форматирования, чтобы LLM не воспринимал его как команды для форматирования. 
    Убирает **, *, `, ###, * в начале строк и * в конце строк.
    '''
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
    '''
    Вызывает LLM модель через OpenAI клиент. Возвращает объект ответа.
    '''
    client = OpenAI(
        api_key=api_key,
        base_url=api_url,
    )
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return response


def _extract_usage_from_response(response):
    '''
    Извлекает токены из ответа LLM, если доступно.
    '''
    usage = {}
    if response is None:
        return usage

    if hasattr(response, 'usage'):
        usage_obj = response.usage
        if usage_obj is not None:
            if isinstance(usage_obj, dict):
                usage = {k: usage_obj.get(k) for k in ['prompt_tokens', 'completion_tokens', 'total_tokens'] if k in usage_obj}
            else:
                usage = {
                    'prompt_tokens': getattr(usage_obj, 'prompt_tokens', None),
                    'completion_tokens': getattr(usage_obj, 'completion_tokens', None),
                    'total_tokens': getattr(usage_obj, 'total_tokens', None),
                }
    return usage


def _safe_extract_content(response):
    '''
    Безопасно получает текст из ответа модели, если структура ответа неожиданная.
    '''
    if response is None:
        return ''

    try:
        return response.choices[0].message.content or ''
    except (AttributeError, IndexError, TypeError):
        return ''


@backoff.on_exception(backoff.expo, (RateLimitError, APITimeoutError, APIError), max_time=90, raise_on_giveup=False, logger=None, on_backoff=backoff_handler)
def get_response_from_llm(
        msg,
        print_debug=False,
        msg_history=None,
        temperature=0.7,
        rate_limit=OPENROUTER_API_RATE_LIMIT_SEC,
        return_usage=False,
        stage=None,
        max_attempts=2,
):
    '''
    Основная функция для получения ответа от LLM. Принимает сообщение от пользователя, историю сообщений, температуру и rate limit. 
    Пытается вызвать несколько API LLM по очереди, пока не получит ответ или не исчерпает все варианты. 
    Логирует ответы и ошибки.
    '''
    if msg_history is None:
        msg_history = []

    new_msg_history = msg_history + [{"role": "user", "content": msg}]
    last_error = None

    for api_idx, (api_key, api_url, api_model) in enumerate(OPENROUTER_APIS):

        if API_DAILY_LIMIT_EXHAUSTED[api_idx]:
            continue

        for attempt in range(1, max_attempts + 1):
            try:
                response = call_llm_model(
                    api_key,
                    api_url,
                    api_model,
                    new_msg_history,
                    temperature
                )
            except JSONDecodeError as e:
                logger.error(f"Ошибка разбора JSON-ответа от API {api_url}: {e}")
                last_error = e
                if attempt < max_attempts:
                    logger.info(f"Повторная попытка запроса к API {api_url} ({attempt+1}/{max_attempts})")
                    time.sleep(rate_limit)
                    continue
                break
            except (RateLimitError, APITimeoutError, APIError):
                raise
            except Exception as e:
                logger.error(f"Непредвиденная ошибка при вызове API {api_url}: {e}")
                last_error = e
                if attempt < max_attempts:
                    time.sleep(rate_limit)
                    continue
                break

            content = _safe_extract_content(response)
            if not content.strip():
                logger.warning(f"Пустой или невалидный ответ от LLM (api={api_url}, model={api_model}) на попытке {attempt}/{max_attempts}")
                last_error = RuntimeError(f"Empty response from {api_url} on attempt {attempt}")
                if attempt < max_attempts:
                    time.sleep(rate_limit)
                    continue
                break

            content = clean_text(content)
            new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]

            usage = _extract_usage_from_response(response)
            if usage is None:
                usage = {}
            if stage:
                record_token_usage(stage, usage)

            logger.info(f'Получен ответ от LLM (idx={api_idx+1}, api={api_url}, model={api_model}):\n {"-"*50}\n {content}\n {"-"*50}\n')

            if print_debug:
                logger.info("*" * 20 + " LLM START " + "*" * 20)
                for j, m in enumerate(new_msg_history):
                    preview = m["content"].replace('\n', ' ')
                    logger.info(f'{j}, {m["role"]}: {preview}...')
                logger.info(content)
                logger.info("*" * 21 + " LLM END " + "*" * 21)

            time.sleep(rate_limit)

            if return_usage:
                return content, new_msg_history, usage
            return content, new_msg_history
        
        logger.info(f"Переход к следующему API после неудачных попыток с {api_url}")
        continue

    if last_error:
        if isinstance(last_error, (RateLimitError, APITimeoutError, APIError)):
            raise last_error
        raise RuntimeError("Не удалось получить валидный ответ от LLM после нескольких попыток") from last_error

    logger.error("Ни один LLM API не был вызван\n")
    raise RuntimeError("Ни один LLM API не был вызван\n")
