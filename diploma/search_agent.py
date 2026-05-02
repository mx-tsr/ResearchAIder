from datetime import datetime, timedelta
import hashlib
import os
import time
import json
import re
import requests
from dataclasses import dataclass, asdict
from typing import List, Optional
import xml.etree.ElementTree as ET

from llm import get_response_from_llm
from utils import load_logger


logger = load_logger()


ARXIV_API_URL = 'http://export.arxiv.org/api/query'  # базовый URL для запросов к arXiv API
ARXIV_API_RATE_LIMIT_SEC = 3.0  # документация arxiv просит делать не чаще 1 запроса в 3 секунды
CACHE_FILE = 'cache/arxiv_cache.json'  # файл для кэширования результатов запросов к arXiv
CACHE_ARXIV_TTL = 24  # время, через которое будет обновляться кэш обращений к arxiv


@dataclass
class ArxivPaper:
    '''
    Класс для хранения информации о статье arXiv. Поля:
    - arxiv_id: str - уникальный идентификатор статьи на arXiv
    - title: str - название статьи
    - abstract: str - аннотация статьи
    - authors: List[str] - список авторов статьи
    - published: str - дата публикации статьи
    - updated: str - дата последнего обновления статьи
    - doi: Optional[str] - DOI статьи, если есть
    - pdf_url: str - URL для скачивания PDF статьи
    - source_url: str - URL страницы статьи на arXiv
    '''
    arxiv_id: str
    title: str
    abstract: str
    authors: List[str]
    published: str
    updated: str
    doi: Optional[str]
    pdf_url: str
    source_url: str


def arxiv_paper_from_dict(data):
    '''
    Переводит информацию о статье из формата словаря в ArxivPaper
    '''
    return ArxivPaper(
        arxiv_id=data.get('arxiv_id', ''),
        title=data.get('title', ''),
        abstract=data.get('abstract', ''),
        authors=data.get('authors', []),
        published=data.get('published', ''),
        updated=data.get('updated', ''),
        doi=data.get('doi'),
        pdf_url=data.get('pdf_url', ''),
        source_url=data.get('source_url', ''),
    )


def rate_limit_sleep(extra=0.2):
    '''
    Выдерживает паузу при обращении к API
    '''
    time.sleep(ARXIV_API_RATE_LIMIT_SEC + extra)


def get_topic_hash(topic):
    '''
    Возвращает хеш темы при кэшировании
    '''
    return hashlib.md5(topic.lower().strip().encode()).hexdigest()


def load_cache():
    '''
    Извлекает результат запросов для темы из кэша
    '''
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_cache(cache):
    '''
    Заносит результат запросов для темы в кэш
    '''
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)


def get_arxiv_id_from_url(arxiv_id_url):
    '''
    Извлекает из url статьи ее arxiv_id. 
    
    arxiv_id_url обычно выглядит так: http://arxiv.org/abs/1234.5678v2, нужно оставить только последнюю часть
    '''
    match = re.search(r'(?:abs|pdf)/([^/]+)', arxiv_id_url)
    if match:
        return match.group(1)
    return arxiv_id_url.strip()


def parse_arxiv_atom(xml):
    '''
    Парсит XML, который возвращает arXiv API, и извлекает из него информацию о статьях.
    
    Возвращает список объектов ArxivPaper.
    '''
    root = ET.fromstring(xml)
    namespaces = {'atom': 'http://www.w3.org/2005/Atom', 'arxiv': 'http://arxiv.org/schemas/atom'}
    papers = []
    for entry in root.findall('atom:entry', namespaces):
        title = entry.findtext('atom:title', default='', namespaces=namespaces).strip()
        abstract = entry.findtext('atom:summary', default='', namespaces=namespaces).strip()
        arxiv_id_url = entry.findtext('atom:id', default='', namespaces=namespaces)
        arxiv_id = get_arxiv_id_from_url(arxiv_id_url)
        authors = [a.findtext('atom:name', namespaces=namespaces).strip() for a in entry.findall('atom:author', namespaces) if a.findtext('atom:name', namespaces=namespaces)]
        published = entry.findtext('atom:published', default='', namespaces=namespaces)
        updated = entry.findtext('atom:updated', default='', namespaces=namespaces)
        doi = entry.findtext('arxiv:doi', default=None, namespaces=namespaces)

        pdf_url = None
        source_url = None
        for link in entry.findall('atom:link', namespaces):
            if link.get('title') == 'pdf':
                pdf_url = link.get('href')
            if link.get('rel') == 'alternate':
                source_url = link.get('href')

        papers.append(ArxivPaper(
            arxiv_id=arxiv_id,
            title=title,
            abstract=abstract,
            authors=authors,
            published=published,
            updated=updated,
            doi=doi,
            pdf_url=pdf_url,
            source_url=source_url or arxiv_id_url,
        ))
    return papers


def format_search_query(topic, field='all'):
    '''
    Преобразует текстовый запрос в формат для поиска arxiv API: 
    добавляет к запросу модификатор поиска и меняет все пробелы на плюсы, удаляет кавычки
    '''
    query = topic.replace('"', '')
    query = re.sub(r"\s+", "+", query)
    return f"{field}:{query}"


def expand_topic_queries(topic, max_number_of_query_variants=7):
    '''
    Расширяет исходную тему, заданную пользователем, генерируя несколько похожих для семантического поиска
    '''
    expanded_queries = []
    expanded_queries.append(format_search_query(topic, 'all'))

    msg, _ = get_response_from_llm(
        expand_topic_prompt.format(
            topic=topic, 
            max_number_of_query_variants=max_number_of_query_variants - 1
            ),
        print_debug=False,
        msg_history=None,
        temperature=0.1
    )

    expanded_queries.extend(map(format_search_query, msg.rstrip().split('\n')[:max_number_of_query_variants - 1]))
    return expanded_queries
    

def search_arxiv(query, max_results=20, start=0, sort_by='relevance', sort_order='descending'):
    '''
    Делает GET-запрос к arxiv API по запросу и возвращает список ArxivPaper 
    '''
    params = {
        'search_query': query,
        'start': start,
        'max_results': max_results,
        'sortBy': sort_by,
        'sortOrder': sort_order,
    }

    response = requests.get(ARXIV_API_URL, params=params, timeout=30)
    if response.status_code != 200:
        raise RuntimeError(f'[ERROR] arXiv API вернуло статус {response.status_code}: {response.text[:400]}')

    papers = parse_arxiv_atom(response.content)
    rate_limit_sleep()
    return papers


def get_set_number_of_papers(topic, num_of_selected_papers=10, total_num_of_papers=70, num_of_expanded_queries=7, papers_per_query=10, store_path=None):
    '''
    Возвращает заданное количество статей по теме. Сначала расширяет тему, потом ищет статьи по каждому расширенному запросу, затем выбирает из них заданное количество наиболее релевантных.
    '''
    logger.info(f'Начинаю поиск заданного числа статей\n')
    logger.info(f'Проверяю нет ли исходной темы в кэше\n')
    
    cache = load_cache()
    topic_hash = get_topic_hash(topic)
    now = datetime.now()

    if topic_hash in cache:
        cached_time = datetime.fromtimestamp(cache[topic_hash]['timestamp'])
        if now - cached_time < timedelta(hours=CACHE_ARXIV_TTL):
            selected_papers = [arxiv_paper_from_dict(p) for p in cache[topic_hash]['selected_papers']]
            logger.info(f'Найден кэш для темы, полученный в течение последних {CACHE_ARXIV_TTL} часов, возвращаю его:\n {[p.arxiv_id for p in selected_papers][:1]} ...\n')
            return selected_papers
        else:
            logger.info(f'Найден кэш для темы, но он был получен более {CACHE_ARXIV_TTL} часов назад. Обновляю его.\n') 
    else:
        logger.info(f'Кэш для темы не найден.\n')
    
    logger.info(f'Начинаю расширение исходной темы\n')
    
    queries = expand_topic_queries(topic, max_number_of_query_variants=num_of_expanded_queries)
    seen = {}

    logger.info(f'Расширенные запросы:\n {"\n".join(queries)}\n')
    search_log = {'topic': topic, 'queries': [], 'selected_papers': []}

    for query in queries:
        logger.info(f'Начинаю поиск статей по запросу: {query}\n')

        entries = search_arxiv(query, max_results=papers_per_query)

        logger.info(f'Нашел всего {len(entries)} статей \n')
        search_log['queries'].append({'query': query, 'found': len(entries), 'papers': [asdict(p) for p in entries]})

        for paper in entries:
            if paper.arxiv_id not in seen:
                seen[paper.arxiv_id] = paper
            if len(seen) >= total_num_of_papers:
                break

        if len(seen) >= total_num_of_papers:
            break

    all_papers = list(seen.values())
    logger.info(f'Всего найдено {len(all_papers)} уникальных статей\n')

    papers_titles_and_abstracts = "\n\n".join([
        f"ID: {p.arxiv_id}\nTitle: {p.title}\nAbstract: {p.abstract[:500]}..." 
        for p in all_papers])

    try:
        top_papers, _ = get_response_from_llm(
            select_top_papers_prompt.format(
                topic=topic, 
                top_k=num_of_selected_papers,
                papers_text=papers_titles_and_abstracts
            ),
            print_debug=False,
            msg_history=None,
            temperature=0.1
        )

        selected_ids = top_papers.rstrip().split('\n')
        selected_ids = [id.strip() for id in selected_ids if id.strip()][:num_of_selected_papers]

        selected_papers = [seen.get(id) for id in selected_ids if id in seen]
        selected_papers = [p for p in selected_papers if p is not None]

        logger.info(f'LLM выбрала {len(selected_papers)} arxiv_id статей:\n {[p.arxiv_id for p in selected_papers]}\n')

    except Exception as e:
        logger.error(f'LLM не смогла выбрать статьи: {e}. Берутся первые {num_of_selected_papers} статей\n')
        selected_papers = all_papers[:num_of_selected_papers]

    # Если LLM выбрал меньше, дополняем первыми из списка
    while len(selected_papers) < num_of_selected_papers and all_papers:
        next_paper = next((p for p in all_papers if p not in selected_papers), None)
        if next_paper:
            selected_papers.append(next_paper)
    
    selected_papers = selected_papers[:num_of_selected_papers]

    for p in selected_papers:
        search_log['selected_papers'].append(asdict(p))

    # сохранение логов поиска
    if store_path:
        with open(store_path, 'w', encoding='utf-8') as f:
            f.write(f'Лог поиска для темы "{topic}", запущен в {time.strftime("%Y-%m-%d %H:%M:%S")}\n\n')
            json.dump(search_log, f, ensure_ascii=False, indent=2)
    
    cache[topic_hash] = {
        'timestamp': now.timestamp(),
        'queries': queries,
        'papers': [asdict(p) for p in all_papers],
        'selected_papers': [asdict(p) for p in selected_papers]
    }
    save_cache(cache)
    logger.info(f'Результат поиска для темы занесен в кеш\n')

    logger.info(f'Поиск статей выполнен успешно. Выбраны {len(selected_papers)} статей\n')

    return selected_papers


expand_topic_prompt = ''' 
Ты — помощник для семантического поиска научных статей на arXiv.
Тебе дана тема научной статьи, по которой нужно осуществить поиск на arXiv: "{topic}".
Твоя задача сформулировать {max_number_of_query_variants} вариантов запроса для поиска похожих и релевантных статей к исходной.
Сформулированные тобой варианты должны быть тесно связаны с исходной темой. Обязательно используй разные формулировки, отличные друг от друга и от исхожной темы. 
Используй искомую тему, синонимы, возможные области приминения, конкретные термины. Каждая тема должны быть формулирована в виде короткой фразы, не более 5-7 слов. Не используй сложные предложения, только ключевые слова и фразы.
Генерируй темы на английском языке.
Обязательно предоставь результат в виде списка, где каждая тема находится на новой строке. Не пиши никакого дополнительного текста, только список тем, без нумерации, пунктов и прочего форматирования. Общее количество тем должно быть ровно {max_number_of_query_variants}.
'''

select_top_papers_prompt = '''
Тебе дана тема: "{topic}"
Выбери топ {top_k} самых релевантных статей из следующего списка, основываясь на их названиях и аннотациях.
Учитывай релевантность теме, новизну и научную ценность.
Возвращай ТОЛЬКО arxiv_ids выбранных тобой статей, по одному на каждой строке, в порядке релевантности (самые релевантные первыми). Сохраняй в arxiv_id приписку версии, если она есть (например, 1234.5678v2), так как она важна для идентификации статьи. 
Не пиши никакого дополнительного текста, только список IDs, без нумерации, пунктов и прочего форматирования и поясняющего текста. 

Статьи для выбора:
{papers_text}
'''


# if __name__ == "__main__":
#     # Тестирую поиск и отбор релевантных статей
#     test_topic = 'multiagent systems of science automation'
#     selected_papers = get_set_number_of_papers(test_topic, num_of_selected_papers=10, total_num_of_papers=70, num_of_expanded_queries=7, papers_per_query=10, store_path='logs/test_arxiv_search_log.json')
#     print(f'Выбранные статьи по теме "{test_topic}": ')
#     for paper in selected_papers:
#         print(f' - {paper["arxiv_id"]}:\n {paper["title"]}\n\nAbstract: {paper["abstract"]}\n\n')
