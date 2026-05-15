import re
import json
import shutil 
import logging
from pathlib import Path
from typing import Dict
from datetime import datetime


def load_logger():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/app.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = load_logger()


def extract_json_from_response(response):
    '''
    Извлекает JSON из ответа LLM.
    Ищет JSON между ```json и ``` или просто JSON объект.
    '''
    # Ищем JSON между маркерами
    json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        # Ищем просто JSON объект
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            return None

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        logger.error(f"Ошибка декодирования JSON: {json_str}")
        return None


def extract_response_block(response, label='RESPONSE'):
    '''
    Извлекает содержимое блока RESPONSE из ответа LLM.
    '''
    patterns = [
        rf'{label}:\s*```(?:\w*)\n(.*?)\n```',
        rf'{label}:\s*`\s*\n(.*?)\n`',
        rf'{label}:\s*(.*)',
    ]
    for pattern in patterns:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()

    if f'{label}:' in response:
        logger.warning(f'Блок {label} найден, но не удалось распарсить формат. Использую остаток после метки\n')
        return response.split(f'{label}:', 1)[1].strip()

    logger.error(f'Не найден блок {label} в ответе\n')
    return None


def load_extracted_info(papers, extracted_info_dir='extracted_info'):
    '''
    Загружает JSON файлы статей из extracted_info и возвращает список словарей.
    Пропускает файлы, в которых отсутствуют обязательные поля.
    '''
    required_fields = ['title', 'problem', 'compared_baselines', 'formulas', 'results', 'limitations', 
                       'novelty', 'key_findings', 'open_questions', 'implicit_gaps', 'arxiv_id']
    papers_info = []
    
    for paper in papers:
        json_path = Path(extracted_info_dir) / f"{paper.arxiv_id}.json"
        if not json_path.exists():
            logger.warning(f"JSON для {paper.arxiv_id} не найден\n")
            continue

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                info = json.load(f)
        except json.JSONDecodeError:
            logger.error(f"Ошибка при чтении JSON для {paper.arxiv_id}\n")
            continue
        
        # Проверяем наличие всех обязательных полей
        missing_fields = []
        for field in required_fields:
            if field not in info or not info[field]:
                missing_fields.append(field)
        
        if missing_fields:
            logger.warning(f"Пропускаю {paper.arxiv_id}: отсутствуют поля {missing_fields}\n")
            continue
        
        info['arxiv_id'] = paper.arxiv_id 
        papers_info.append(info)
    
    return papers_info


def clear_directory(path):
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_to_current(source_path, current_dir):
    if not current_dir:
        return
    current_dir = Path(current_dir)
    current_dir.mkdir(parents=True, exist_ok=True)
    destination = current_dir / Path(source_path).name
    shutil.copy2(source_path, destination)
    return str(destination)


TOKEN_USAGE_TRACKING = None
DEFAULT_TOKEN_TRACKING_STAGES = ['search_agent', 'extraction_agent', 'analysis_agent', 'writeup_agent']


def init_token_usage_tracking(stages=None):
    """
    Инициализирует глобальный трекер токенов по агентам
    """
    global TOKEN_USAGE_TRACKING
    if stages is None:
        stages = DEFAULT_TOKEN_TRACKING_STAGES

    TOKEN_USAGE_TRACKING = {
        stage: {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0,
            'calls': 0,
        }
        for stage in stages
    }
    TOKEN_USAGE_TRACKING['total'] = {
        'prompt_tokens': 0,
        'completion_tokens': 0,
        'total_tokens': 0,
        'calls': 0,
    }
    return TOKEN_USAGE_TRACKING


def record_token_usage(stage, usage):
    """
    Сохраняет данные о затраченных токенах для указанного этапа
    """
    global TOKEN_USAGE_TRACKING
    if TOKEN_USAGE_TRACKING is None:
        init_token_usage_tracking()

    if stage not in TOKEN_USAGE_TRACKING:
        TOKEN_USAGE_TRACKING[stage] = {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0,
            'calls': 0,
        }

    prompt_tokens = usage.get('prompt_tokens', 0)
    completion_tokens = usage.get('completion_tokens', 0)
    total_tokens = usage.get('total_tokens', 0)
    if total_tokens == 0:
        total_tokens = prompt_tokens + completion_tokens

    TOKEN_USAGE_TRACKING[stage]['prompt_tokens'] += prompt_tokens
    TOKEN_USAGE_TRACKING[stage]['completion_tokens'] += completion_tokens
    TOKEN_USAGE_TRACKING[stage]['total_tokens'] += total_tokens
    TOKEN_USAGE_TRACKING[stage]['calls'] += 1

    TOKEN_USAGE_TRACKING['total']['prompt_tokens'] += prompt_tokens
    TOKEN_USAGE_TRACKING['total']['completion_tokens'] += completion_tokens
    TOKEN_USAGE_TRACKING['total']['total_tokens'] += total_tokens
    TOKEN_USAGE_TRACKING['total']['calls'] += 1


def get_token_usage_summary():
    """
    Возвращает текущую статистику использования токенов
    """
    global TOKEN_USAGE_TRACKING
    if TOKEN_USAGE_TRACKING is None:
        init_token_usage_tracking()
    return {stage: dict(data) for stage, data in TOKEN_USAGE_TRACKING.items()}


def save_token_usage(topic, output_dir='analysis_output'):
    """
    Сохраняет информацию об использовании токенов в JSON-файл
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    token_file = output_path / 'token_usage_tracking.json'

    if token_file.exists():
        try:
            with open(token_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = {
                'experiment': 'Token Usage Tracking',
                'timestamp': datetime.now().isoformat(),
                'topics': []
            }
    else:
        data = {
            'experiment': 'Token Usage Tracking',
            'timestamp': datetime.now().isoformat(),
            'topics': []
        }

    summary = get_token_usage_summary()
    topic_data = {
        'topic': topic,
        'prompt_tokens_total': summary['total']['prompt_tokens'],
        'completion_tokens_total': summary['total']['completion_tokens'],
        'tokens_total': summary['total']['total_tokens'],
        'prompt_completion_sum': summary['total']['prompt_tokens'] + summary['total']['completion_tokens'],
        'total_calls': summary['total']['calls'],
        'token_usage_by_stage': summary,
        'recorded_at': datetime.now().isoformat()
    }

    data['topics'].append(topic_data)

    with open(token_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info(f'Информация об использовании токенов сохранена в {token_file}\n')
