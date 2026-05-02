import re
import json
import logging
from pathlib import Path


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
    required_fields = ['title', 'problem', 'compared_baselines', 'results', 'limitations', 
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