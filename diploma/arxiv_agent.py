
import os
import time
import json
import re
import requests
from dataclasses import dataclass, asdict
from typing import List, Optional
from pathlib import Path
import xml.etree.ElementTree as ET

ARXIV_API_URL = 'http://export.arxiv.org/api/query'  # базовый URL для запросов к arXiv API
ARXIV_API_RATE_LIMIT_SEC = 3.0  # документация просит делать не чаще 1 запроса в 3 секунды


@dataclass
class ArxivPaper:
    arxiv_id: str
    title: str
    abstract: str
    authors: List[str]
    published: str
    updated: str
    doi: Optional[str]
    pdf_url: str
    source_url: str


def _rate_limit_sleep(extra=0.2):
    time.sleep(ARXIV_API_RATE_LIMIT_SEC + extra)


def get_arxiv_id_from_url(arxiv_id_url):
    # arxiv_id_url обычно выглядит так: http://arxiv.org/abs/1234.5678v2, нужно оставить только последнюю часть
    match = re.search(r'(?:abs|pdf)/([^/]+)', arxiv_id_url)
    if match:
        return match.group(1)
    return arxiv_id_url.strip()


def parse_arxiv_atom(xml):
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
    # добавляет к запросу модификатор поиска и меняет все пробелы на плюсы, удаляет кавычки
    query = topic.replace('"', '')
    query = re.sub(r"\s+", "+", query)
    return f"{field}:{query}"


def expand_topic_queries(topic, extra_context=None, max_number_of_query_variants=5):
    expanded_queries = []

    # 1. Изначальный запрос пользователя
    expanded_queries.append(format_search_query(topic, 'all'))

    # 2. Поиск по заголовкам
    expanded_queries.append(format_search_query(topic, 'ti'))

    # 3. Поиск по аннотациям
    expanded_queries.append(format_search_query(topic, 'abs'))

    # 4. Поиск по ключевым словам
    default_terms = ['agentic system', 'science automation', 'research assistant', 'AI research agent', 'autonomous research']
    if extra_context:
        default_terms = extra_context + default_terms

    for t in default_terms:
        if len(expanded_queries) >= max_number_of_query_variants:
            break
        if t.lower() in topic.lower():
            continue
        expanded_queries.append(format_search_query(f'{topic} {t}'))

    unique_expanded_queries = []
    for q in expanded_queries:
        if q not in unique_expanded_queries:
            unique_expanded_queries.append(q)
        if len(unique_expanded_queries) >= max_number_of_query_variants:
            break
    return unique_expanded_queries


def search_arxiv(query, max_results=20, start=0, sort_by='relevance', sort_order='descending'):
    params = {
        'search_query': query,
        'start': start,
        'max_results': max_results,
        'sortBy': sort_by,
        'sortOrder': sort_order,
    }

    response = requests.get(ARXIV_API_URL, params=params, timeout=30)
    if response.status_code != 200:
        raise RuntimeError(f'arXiv API returned status {response.status_code}: {response.text[:400]}')

    papers = parse_arxiv_atom(response.content)
    _rate_limit_sleep()
    return papers


def get_set_number_of_papers(topic, num_of_papers=10, num_of_expanded_queries=5, papers_per_query=20, store_path=None):
    queries = expand_topic_queries(topic, max_number_of_query_variants=num_of_expanded_queries)
    seen = {}

    search_log = {'topic': topic, 'queries': [], 'papers': []}

    for query in queries:
        entries = search_arxiv(query, max_results=papers_per_query)
        search_log['queries'].append({'query': query, 'found': len(entries)})

        for paper in entries:
            if paper.arxiv_id not in seen:
                seen[paper.arxiv_id] = paper
            if len(seen) >= num_of_papers:
                break

        if len(seen) >= num_of_papers:
            break

    result = list(seen.values())[:num_of_papers]

    for p in result:
        search_log['papers'].append(asdict(p))

    # сохранение логов поиска
    if store_path:
        with open(store_path, 'w', encoding='utf-8') as f:
            json.dump(search_log, f, ensure_ascii=False, indent=2)

    return result


def download_pdf(arxiv_id_url, target_dir='pdfs', timeout=120):
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    arxiv_id = get_arxiv_id_from_url(arxiv_id_url)
    if arxiv_id.endswith("v"):
        arxiv_id = arxiv_id[:-1]

    pdf_url = f'https://arxiv.org/pdf/{arxiv_id}.pdf'
    out_path = target_dir / f'{arxiv_id}.pdf'

    #  если файл уже существует и его размер больше 1KB, считаем, что он уже скачан
    if out_path.exists() and out_path.stat().st_size > 1024:
        return str(out_path)

    response = requests.get(pdf_url, stream=True, timeout=timeout)
    if response.status_code != 200:
        raise RuntimeError(f'Failed download {pdf_url}: {response.status_code}')

    with open(out_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    # после запроса выдерживаем ограичение в 3 секунды
    _rate_limit_sleep()

    return str(out_path)


def download_papers(papers, pdf_dir='pdfs'):
    downloaded_files = []
    for paper in papers:
        try:
            path = download_pdf(paper.arxiv_id, target_dir=pdf_dir)
            downloaded_files.append(path)
        except Exception as ex:
            print(f'[WARNING] Could not download {paper.arxiv_id}: {ex}')
    return downloaded_files


def search_and_download_arxiv_papers(topic, num_of_papers=10, num_of_expanded_queries=5, store_results='arxiv_search_log.json'):
    """
    Основная функция: 
    1. Расширение запросов (пока вместо семантического поиска), 
    2. Поиск статей, 
    3. Выбор 10, 
    4. Скачивание PDF
    """
    papers = get_set_number_of_papers(topic, num_of_papers=num_of_papers, num_of_expanded_queries=num_of_expanded_queries, store_path=store_results)
    pdf_paths = download_papers(papers, pdf_dir='pdfs')
    return {
        'topic': topic,
        'selected_papers': [asdict(p) for p in papers],
        'pdf_paths': pdf_paths,
        'log_file': store_results,
    }
