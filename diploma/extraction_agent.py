import json
import re
from pathlib import Path
import pymupdf4llm
from PyPDF2 import PdfReader
import pymupdf
import requests

from search_agent import ArxivPaper, rate_limit_sleep
from llm import get_response_from_llm
from utils import load_logger, extract_json_from_response

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


def download_papers(papers, pdf_dir='pdfs'):
    '''
    Скачивает статьи в формате pdf по их arxiv_id и сохраняет в pdf_dir. Возвращает список путей к сохраненным файлам.
    '''
    downloaded_files = []
    for paper in papers:
        try:
            path = download_pdf(paper.arxiv_id, target_dir=pdf_dir)
            downloaded_files.append(path)
        except Exception as e:
            logger.error(f'Не смог скачать {paper.arxiv_id}: {e}\n')
    return downloaded_files


def format_pdf_text(text):
    '''
    Форматирует извлеченный из PDF текст, оставляя только основное содержание статьи и удаляя ненужные разделы.
    '''
    # Убираем ** с обеих сторон
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)

    # Оставляем текст между Introduction и Acknowledgements
    start_of_text = text.lower().find('introduction')
    end_of_text = None

    # Сначала обрезаем по Приложению, если возможно
    if text.lower().find('## appendix') != -1:
        end_of_text = text.lower().find('## appendix')
    elif text.lower().find('appendix') != -1:
        end_of_text = text.lower().find('appendix')
    if end_of_text:
        text = text[:end_of_text] 
    # Затем обрезаем по содержанию
    if text.lower().find('acknowledgements') != -1:
        end_of_text = text.lower().find('acknowledgements')
    elif text.lower().find('acknowledgement') != -1:
        end_of_text = text.lower().find('acknowledgement')
    elif text.lower().find('acknowledgments') != -1:
        end_of_text = text.lower().find('acknowledgments')
    elif text.lower().find('acknowledgment') != -1:
        end_of_text = text.lower().find('acknowledgment')
    elif text.lower().find('## references') != -1:
        end_of_text = text.lower().find('## references')
    elif text.lower().find('references') != -1:
        end_of_text = text.lower().find('references')
    
    if end_of_text:
        text = text[:end_of_text] 
    text = text[start_of_text:]

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


def extract_and_save_texts(papers, txt_dir='txts', pdf_dir='pdfs'):
    '''
    Извлекает текст из PDF статей и сохраняет его в txt файлы. Возвращает список путей к сохраненным txt файлам.'''
    txt_dir_path = Path(txt_dir)
    txt_dir_path.mkdir(parents=True, exist_ok=True)
    
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
            continue
        
        text = extract_text_from_paper_pdf(str(pdf_path))
        cleaned_text = format_pdf_text(text)
        
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(paper.title + "\n\n")
            f.write(cleaned_text)
        
        txt_paths.append(str(txt_path))
        logger.info(f"Текст для {paper.arxiv_id} сохранён в {txt_path}\n")
    
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
                rate_limit=OPENROUTER_API_EXTRACT_RATE_LIMIT_SEC
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
                msg_history=msg_history
            )
            
            if "ПРИНЯТО" in response.upper():
                break
            else:
                # Попытка извлечь исправленный JSON
                corrected = extract_json_from_response(response)
                if corrected:
                    extracted = corrected

    return extracted


def extract_key_info_from_papers(papers, output_dir='extracted_info', txt_dir='txts'):
    '''
    Обрабатывает все txt файлы статей в директории и сохраняет извлеченную информацию в JSON.
    '''
    extracted_info = []

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    for paper in papers:
        txt_path = Path(txt_dir) / f"{paper.arxiv_id}.txt"
        if not txt_path.exists():
            logger.error(f"TXT для {paper.arxiv_id} не найден\n")
            continue

        output_path = output_dir_path / f"{paper.arxiv_id}.json"

        if output_path.exists() and output_path.stat().st_size > 1024:
            logger.info(f"json для {paper.arxiv_id} уже существует, пропускаю извлечение\n")
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
        else:
            logger.error(f"Не удалось извлечь информацию из {txt_path}\n")
    
    return extracted_info


def download_and_extract_arxiv_papers(papers, pdf_dir='pdfs', txt_dir='txts', extracted_info_dir='extracted_info'):
    """
    Скачивание PDF отобранных статей, извлекает из них текст в txt и визвлекает из них ключевую информацию в формате JSON
    """
    
    logger.info(f'Начинаю скачивание PDF статей\n')
    
    pdf_paths = download_papers(papers, pdf_dir=pdf_dir)

    logger.info(f'Скачано {len(pdf_paths)} PDF статей. Начинаю извлечение текста из PDF\n')

    txt_paths = extract_and_save_texts(papers, txt_dir=txt_dir, pdf_dir='pdfs')

    logger.info(f'Извлечено {len(txt_paths)} txt статей. Начинаю извлечение ключевой информации из txt\n')

    extracted_info_paths = extract_key_info_from_papers(papers, output_dir=extracted_info_dir, txt_dir=txt_dir)

    logger.info(f'Извлечена ключевая информация из {len(extracted_info_paths)} статей\n')

    return extracted_info_paths
    

# Промпт для извлечения информации из статьи
extraction_prompt = """
Тебе дан текст научной статьи. Проанализируй его и извлеки следующие поля в формате JSON:

- title: Название статьи
- problem: Главная проблема или исследовательский вопрос
- compared_baselines: Какие методы или работы сравниваются в статье и как они соотносятся с предлагаемым подходом
- results: Ключевые результаты или открытия
- limitations: Любые ограничения, упомянутые в статье
- novelty: Что новое или уникальное в этой работе
- key_findings: 1-2 предложения, резюмирующие основные результаты
- open_questions: Нерешенные вопросы или будущая работа, упомянутая в статье
- implicit_gaps: Любые пробелы в исследованиях, которые неявно видны из текста, даже если авторы их не выделяют

Отвечай только объектом JSON, без дополнительного текста. Объект JSON выделяется таким блоком: ```json  ```

Текст статьи:
=== НАЧАЛО ТЕКСТА СТАТЬИ ===
{text}
=== КОНЕЦ ТЕКСТА СТАТЬИ ===
"""


# Промпт для проверки качества извлечения
extraction_quality_check_prompt = """
Оцени извлеченную информацию из статьи. Проверьте точность, полноту и согласованность с оригинальным текстом.

Извлеченная информация:
=== НАЧАЛО ИНФОРМАЦИИ ===
{extracted_json}
=== КОНЕЦ ИНФОРМАЦИИ ===

Дай обратную связь и предложи улучшения. Если информация удовлетворительна, напиши "ПРИНЯТО". В противном случае, предоставь исправленный JSON.
Отвечай либо "ПРИНЯТО", либо исправленным JSON.
"""


if __name__ == "__main__":
    user_topic = 'multiagent systems of science automation'
        
    # Тестирую полный цикл поиска статей, скачивания и извлечения
    # from search_agent import get_set_number_of_papers
    # selected_papers = get_set_number_of_papers(topic=user_topic, num_of_selected_papers=10, total_num_of_papers=70, num_of_expanded_queries=7, papers_per_query=10, store_path='logs/arxiv_search_log.json')
    # extracted_info_from_papers = download_and_extract_arxiv_papers(selected_papers, pdf_dir='pdfs', txt_dir='txts', extracted_info_dir='extracted_info')
    
    # Тестирую удаление ненужной информации из пдф
    # text = extract_text_from_paper_pdf('pdfs/2504.03424v1.pdf')
    # cleaned_text = format_pdf_text(text)
    # print(cleaned_text)