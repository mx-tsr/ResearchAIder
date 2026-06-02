import json
import re
import shutil
from pathlib import Path
import pymupdf4llm
from PyPDF2 import PdfReader
import pymupdf
import requests

from search_agent import ArxivPaper, rate_limit_sleep
from llm import get_response_from_llm
from utils import load_logger, extract_json_from_response, copy_to_current, clear_directory

OPENROUTER_API_EXTRACT_RATE_LIMIT_SEC = 5.0 


logger = load_logger()


def download_pdf(arxiv_id, target_dir='pdfs', timeout=120):
    '''
    Скачивает статью в формате pdf по ее arxiv_id_url и сохраняет в target_dir. Возвращает путь к сохраненному файлу.
    '''
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    pdf_url = f'https://arxiv.org/pdf/{arxiv_id}.pdf'
    out_path = target_dir / f'{arxiv_id}.pdf'

    logger.info(f'Начинаю скачивание PDF для {arxiv_id}\n')

    #  если файл уже существует и его размер больше 1KB, считаем, что он уже скачан
    if out_path.exists() and out_path.stat().st_size > 1024:
        logger.info(f'Файл {arxiv_id} уже существует, пропускаю скачивание\n')

        return str(out_path)

    response = requests.get(pdf_url, stream=True, timeout=timeout)
    if response.status_code != 200:
        logger.error(f'Не удалось скачать {pdf_url}: {response.status_code}')
        raise RuntimeError(f'Не удалось скачать {pdf_url}: {response.status_code}')
    
    logger.info(f'Получен ответ от сервера {response.status_code}. Начинаю запись файла\n')

    with open(out_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    logger.info(f'Файл {arxiv_id} успешно скачан и сохранен в {out_path}\n')

    # после запроса выдерживаем ограничение в 3 секунды
    rate_limit_sleep()

    return str(out_path)


def copy_local_pdf(paper, pdf_dir='pdfs'):
    '''
    Копирует локальный PDF из input_papers в папку pdfs.
    '''
    target_dir = Path(pdf_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    source_path = Path(paper.local_pdf_path)
    target_path = target_dir / f"{paper.arxiv_id}.pdf"

    if not source_path.exists():
        logger.warning(f'Локальный PDF не найден: {source_path}\n')
        return ''

    if target_path.exists() and target_path.stat().st_size > 1024:
        logger.info(f'Файл {target_path} уже существует, пропускаю копирование\n')
        return str(target_path)

    shutil.copy2(source_path, target_path)
    logger.info(f'Скопирован локальный PDF {source_path} в {target_path}\n')
    return str(target_path)


def download_papers(papers, pdf_dir='pdfs'):
    '''
    Скачивает статьи в формате pdf по их arxiv_id и сохраняет в pdf_dir. Возвращает список путей к сохраненным файлам.
    Для локальных PDF копирует их в папку pdfs вместо скачивания.
    '''
    downloaded_files = []
    for paper in papers:
        try:
            if getattr(paper, 'local_pdf_path', None):
                path = copy_local_pdf(paper, pdf_dir=pdf_dir)
            else:
                path = download_pdf(paper.arxiv_id, target_dir=pdf_dir)
            downloaded_files.append(path)
        except Exception as e:
            logger.error(f'Не смог получить PDF для {paper.arxiv_id}: {e}\n')
    return downloaded_files


def format_pdf_text(text):
    '''
    Форматирует извлеченный из PDF текст, оставляя только основное содержание статьи и удаляя ненужные разделы.
    '''
    # Убираем ** с обеих сторон
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)

    # Оставляем текст между Introduction и Acknowledgements
    start_of_text = 0
    if text.lower().find('introduction', 0, 2000) != -1:
        start_of_text = text.lower().find('introduction')
    elif text.lower().find('abstract') != -1:
        start_of_text = text.lower().find('abstract')

    text = text[start_of_text:]

    end_of_text = len(text)

    # Сначала обрезаем по Приложению, если возможно
    if text.lower().find('## appendix') != -1:
        end_of_text = text.lower().find('## appendix')
    if end_of_text != len(text):
        text = text[:end_of_text] 
        
    # Затем обрезаем по содержанию
    if text.lower().find('acknowledgements', 5000) != -1:
        end_of_text = text.lower().find('acknowledgements', 5000)
    elif text.lower().find('acknowledgement', 5000) != -1:
        end_of_text = text.lower().find('acknowledgement', 5000)
    elif text.lower().find('acknowledgments', 5000) != -1:
        end_of_text = text.lower().find('acknowledgments', 5000)
    elif text.lower().find('acknowledgment', 5000) != -1:
        end_of_text = text.lower().find('acknowledgment', 5000)
    elif text.lower().find('## references', 5000) != -1:
        end_of_text = text.lower().find('## references', 5000)

    if end_of_text != len(text):
        text = text[:end_of_text] 

    # Удаляем обозначения картинок
    text = re.sub(r'==> picture.*?<==', '', text, flags=re.DOTALL)
    text = re.sub(r'----- Start of picture text -----.*?----- End of picture text -----', '', text, flags=re.DOTALL)

    # Удаляем обозначения графиков
    pattern = r'^(Figure|Fig\.)\s*\d+.*$'
    text = re.sub(pattern, '', text, flags=re.MULTILINE)

    # Нормализуем пробелы
    text = re.sub(r'\n\s*\n', '\n', text)  # убираем пустые строки
    text = re.sub(r' +', ' ', text)  # множественные пробелы
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text) # ** с двух сторон

    if len(text.split('\n')) > 400:
        text = '\n'.join(text.split('\n')[:400])  # ограничиваем 400 строками

    return text


def extract_text_from_paper_pdf(pdf_path, min_size=100):
    '''
    Извлекает текст из PDF статьи, используя несколько разных библиотек для повышения надежности. Сначала пробует pymupdf4llm, если текст слишком короткий, пробует pymupdf, а если и он не справляется, использует pypdf.
    '''
    try:
        text = pymupdf4llm.to_markdown(pdf_path)
        if len(text) < min_size:
            raise Exception("[ERROR] текст слишком короткий")
    except Exception as e:
        logger.error(f"pymupdf4llm не смог извлечь текст, попытаюсь использовать pymupdf: {e}\n")
        try:
            doc = pymupdf.open(pdf_path)  
            text = ""
            for page in doc:  
                text = text + page.get_text()  
            if len(text) < min_size:
                raise Exception("[ERROR] текст слишком короткий")
        except Exception as e:
            logger.error(f"pymupdf не смог извлечь текст, попытаюсь использовать pypdf: {e}\n")
            reader = PdfReader(pdf_path)
            text = "".join(page.extract_text() for page in reader.pages)
            if len(text) < min_size:
                raise Exception("[ERROR] текст слишком короткий")

    return text


def extract_and_save_texts(papers, txt_dir='txts', pdf_dir='pdfs', current_txt_dir=None):
    '''
    Извлекает текст из PDF статей и сохраняет его в txt файлы. Возвращает список путей к сохраненным txt файлам.'''
    txt_dir_path = Path(txt_dir)
    txt_dir_path.mkdir(parents=True, exist_ok=True)
    if current_txt_dir is not None:
        clear_directory(current_txt_dir)

    txt_paths = []
    for paper in papers:
        pdf_path = Path(pdf_dir) / f"{paper.arxiv_id}.pdf"
        if not pdf_path.exists():
            logger.error(f"PDF для {paper.arxiv_id} не найден\n")
            continue

        txt_path = txt_dir_path / f"{paper.arxiv_id}.txt"
        if txt_path.exists() and txt_path.stat().st_size > 1024:
            logger.info(f"TXT для {paper.arxiv_id} уже существует, пропускаю извлечение\n")
            txt_paths.append(str(txt_path))
            if current_txt_dir is not None:
                copy_to_current(txt_path, current_txt_dir)
            continue
        
        text = extract_text_from_paper_pdf(str(pdf_path))
        cleaned_text = format_pdf_text(text)
        
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(paper.title + "\n\n")
            f.write(cleaned_text)
        
        txt_paths.append(str(txt_path))
        logger.info(f"Текст для {paper.arxiv_id} сохранён в {txt_path}\n")
        if current_txt_dir is not None:
            copy_to_current(txt_path, current_txt_dir)
    
    return txt_paths


def extract_key_info_from_paper(text, num_iterations=3):
    '''
    Извлекает ключевую информацию по полям из текста статьи с итеративной проверкой качества.
    '''
    extracted = None
    msg_history = None

    for iteration in range(num_iterations):
        if iteration == 0:
            # Первичное извлечение
            response, msg_history = get_response_from_llm(
                msg=extraction_prompt.format(text=text),
                print_debug=False,
                rate_limit=OPENROUTER_API_EXTRACT_RATE_LIMIT_SEC,
                stage='extraction_agent'
            )
            extracted = extract_json_from_response(response)
        else:
            # Проверка качества
            if extracted is None:
                logger.error('Не удалось найти json в ответе LLM и извлечь ключевую информацию\n')
                continue
            
            response, msg_history = get_response_from_llm(
                msg=extraction_quality_check_prompt.format(
                    extracted_json=json.dumps(extracted, indent=2)
                ),
                print_debug=False,
                msg_history=msg_history,
                stage='extraction_agent'
            )
            
            if "ПРИНЯТО" in response.upper():
                break
            else:
                # Попытка извлечь исправленный JSON
                corrected = extract_json_from_response(response)
                if corrected:
                    extracted = corrected

    return extracted


def extract_key_info_from_papers(papers, output_dir='extracted_info', txt_dir='txts', current_extracted_info_dir=None):
    '''
    Обрабатывает все txt файлы статей в директории и сохраняет извлеченную информацию в JSON.
    '''
    extracted_info = []

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    if current_extracted_info_dir is not None:
        clear_directory(current_extracted_info_dir)

    for paper in papers:
        txt_path = Path(txt_dir) / f"{paper.arxiv_id}.txt"
        if not txt_path.exists():
            logger.error(f"TXT для {paper.arxiv_id} не найден\n")
            continue

        output_path = output_dir_path / f"{paper.arxiv_id}.json"

        if output_path.exists() and output_path.stat().st_size > 1024:
            logger.info(f"json для {paper.arxiv_id} уже существует, пропускаю извлечение\n")
            if current_extracted_info_dir is not None:
                copy_to_current(output_path, current_extracted_info_dir)
            continue

        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read()

        logger.info(f"Обработка текста статьи {paper.arxiv_id}.txt\n")
        info = extract_key_info_from_paper(text)

        if info:
            info['arxiv_id'] = paper.arxiv_id 
            extracted_info.append(info)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(info, f, indent=2, ensure_ascii=False)

            logger.info(f"Ключевая информация для {paper.arxiv_id} сохранена в {output_path}\n")
            if current_extracted_info_dir is not None:
                copy_to_current(output_path, current_extracted_info_dir)
        else:
            logger.error(f"Не удалось извлечь информацию из {txt_path}\n")
    
    return extracted_info


def download_and_extract_arxiv_papers(papers, pdf_dir='pdfs', txt_dir='txts', extracted_info_dir='extracted_info'):
    """
    Скачивание PDF отобранных статей, извлекает из них текст в txt и извлекает из них ключевую информацию в формате JSON
    """
    pdf_current_dir = Path(pdf_dir) / 'current'
    txt_current_dir = Path(txt_dir) / 'current'
    extracted_info_current_dir = Path(extracted_info_dir) / 'current'

    clear_directory(pdf_current_dir)
    clear_directory(txt_current_dir)
    clear_directory(extracted_info_current_dir)

    logger.info(f'Начинаю скачивание PDF статей\n')
    pdf_paths = download_papers(papers, pdf_dir=pdf_dir)

    for pdf_path in pdf_paths:
        if pdf_path:
            copy_to_current(pdf_path, pdf_current_dir)

    logger.info(f'Скачано {len(pdf_paths)} PDF статей. Начинаю извлечение текста из PDF\n')

    txt_paths = extract_and_save_texts(
        papers,
        txt_dir=txt_dir,
        pdf_dir=pdf_dir,
        current_txt_dir=txt_current_dir
    )

    logger.info(f'Извлечено {len(txt_paths)} txt статей. Начинаю извлечение ключевой информации из txt\n')

    extracted_info_paths = extract_key_info_from_papers(
        papers,
        output_dir=extracted_info_dir,
        txt_dir=txt_dir,
        current_extracted_info_dir=extracted_info_current_dir
    )

    logger.info(f'Извлечена ключевая информация из {len(extracted_info_paths)} статей\n')

    return extracted_info_paths
    

# Промпт для извлечения информации из статьи
extraction_prompt = """
Тебе дан текст научной статьи. Проанализируй его и извлеки следующие поля в формате JSON:

- title: Название статьи
- problem: Главная проблема или исследовательский вопрос
- compared_baselines: Какие методы или работы сравниваются в статье и как они соотносятся с предлагаемым подходом и между собой. 
- formulas: Если есть, описание используемого математического аппарата, формул, их представление и использование, но пиши формулы не в формате latex - это ломает json.
- results: Ключевые результаты или открытия, полученные метрики
- limitations: Любые ограничения, упомянутые в статье
- novelty: Что новое или уникальное в этой работе
- key_findings: 1-2 предложения, резюмирующие основные результаты
- open_questions: Нерешенные вопросы или будущая работа, упомянутая в статье
- implicit_gaps: Любые пробелы в исследованиях, которые неявно видны из текста, даже если авторы их не выделяют

Отвечай только объектом JSON и на русском языке, без дополнительного текста, исключай любые кавычки - это ломает json. Объект JSON выделяется таким блоком: ```json  ```
ОТВЕЧАЙ СТРОГО В JSON ФОРМАТЕ:
{{
    "title": "...",
    "problem": "...",
    "compared_baselines": "...",
    "formulas": "...",
    "results": "...",
    "limitations": "...",
    "novelty": "...",
    "key_findings": "...",
    "open_questions": "...",
    "implicit_gaps": "..."
}}

Текст статьи:
=== НАЧАЛО ТЕКСТА СТАТЬИ ===
{text}
=== КОНЕЦ ТЕКСТА СТАТЬИ ===
"""


# Промпт для проверки качества извлечения
extraction_quality_check_prompt = """
Оцени извлеченную информацию из статьи. Проверьте точность, полноту и согласованность с оригинальным текстом.
При необходимости внеси улучшения в извлеченную информацию, которые повысят её точность и полноту в соответсвии с текстом статьи. Если информация представлена полностью, напиши только "ПРИНЯТО". В противном случае, предоставь улучшенную извлеченную информацию в формате JSON:

{{
    "title": "...",
    "problem": "...",
    "compared_baselines": "...",
    "formulas": "...",
    "results": "...",
    "limitations": "...",
    "novelty": "...",
    "key_findings": "...",
    "open_questions": "...",
    "implicit_gaps": "..."
}}

Отвечай либо "ПРИНЯТО", либо исправленным JSON.

Извлеченная информация:
=== НАЧАЛО ИНФОРМАЦИИ ===
{extracted_json}
=== КОНЕЦ ИНФОРМАЦИИ ===

"""


# if __name__ == "__main__":
#     user_topic = 'multiagent systems of science automation'
        
    # Тестирую полный цикл поиска статей, скачивания и извлечения
    # from search_agent import get_set_number_of_papers
    # selected_papers = get_set_number_of_papers(topic=user_topic, num_of_selected_papers=10, total_num_of_papers=70, num_of_expanded_queries=7, papers_per_query=10, store_path='logs/arxiv_search_log.json')
    # selected_papers = [
    #         ArxivPaper('2505.13400v1', None, None, None, None, None, None, None, None),
    #         ArxivPaper('2512.13930v1', None, None, None, None, None, None, None, None),
    #         ArxivPaper('2508.19383v1', None, None, None, None, None, None, None, None),
    #         ArxivPaper('2511.07262v2', None, None, None, None, None, None, None, None),
    #         ArxivPaper('2505.15047v4', None, None, None, None, None, None, None, None),
    #         ArxivPaper('2508.05666v1', None, None, None, None, None, None, None, None),
    #         ArxivPaper('2502.06472v2', None, None, None, None, None, None, None, None),
    #         ArxivPaper('2601.03794v1', None, None, None, None, None, None, None, None),
    #         ArxivPaper('2509.19326v1', None, None, None, None, None, None, None, None),
    #         ArxivPaper('2011.01103v1', None, None, None, None, None, None, None, None),
    #     ]
    # extracted_info_from_papers = download_and_extract_arxiv_papers(selected_papers, pdf_dir='pdfs', txt_dir='txts', extracted_info_dir='extracted_info')
    
    # Тестирую удаление ненужной информации из пдф
    # text = extract_text_from_paper_pdf('pdfs/2504.03424v1.pdf')
    # cleaned_text = format_pdf_text(text)
    # print(cleaned_text)