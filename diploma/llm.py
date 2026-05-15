import re
import time
import backoff
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


@backoff.on_exception(backoff.expo, (RateLimitError, APITimeoutError, APIError), max_time=90, raise_on_giveup=False, logger=None, on_backoff=backoff_handler)
def get_response_from_llm(
        msg,
        print_debug=False,
        msg_history=None,
        temperature=0.7,
        rate_limit=OPENROUTER_API_RATE_LIMIT_SEC,
        return_usage=False,
        stage=None,
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

        try:
            response = call_llm_model(
                api_key, 
                api_url, 
                api_model, 
                new_msg_history, 
                temperature
            )

            content = response.choices[0].message.content if response is not None else ''
            if content:
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
        
        except RateLimitError as e:
            logger.error(f"Rate limit 429 для API {api_url}: {e}\n")
            last_error = e

            if "Rate limit exceeded: free-models-per-day. Add 10 credits to unlock 1000 free model requests per day" in str(e):
                API_DAILY_LIMIT_EXHAUSTED[api_idx] = True
                logger.info(f"API #{api_idx+1} исчерпал дневной лимит\n")
                continue
            else:
                # Другие RateLimitError обрабатываются через backoff
                raise

        except APITimeoutError as e:
            logger.error(f"API timeout для API {api_idx} {api_url}: {e}\n")
            last_error = e
            API_DAILY_LIMIT_EXHAUSTED[api_idx] = True
            raise

        except APIError as e:
            logger.error(f"API error для API {api_idx} {api_url}: {e}\n")
            if "401" in str(e) or "Unauthorized" in str(e):
                logger.error(f"Проверьте API #{api_idx} в .env\n")
            elif "404" in str(e):
                logger.error(f"Модель '{api_model}' не найдена\n")
            last_error = e
            API_DAILY_LIMIT_EXHAUSTED[api_idx] = True
            raise

    if last_error:
        raise last_error

    logger.error("Ни один LLM API не был вызван\n")
    raise RuntimeError("Ни один LLM API не был вызван\n")
