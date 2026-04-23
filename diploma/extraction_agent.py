import json
import os
import re

from pathlib import Path

from arxiv_agent import ArxivPaper
from llm_agent import get_response_from_llm

OPENROUTER_API_EXTRACT_RATE_LIMIT_SEC = 5.0 

def extract_json_from_response(response):
    """
    Извлекает JSON из ответа LLM.
    Ищет JSON между ```json и ``` или просто JSON объект.
    """
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
        print(f"[ERROR] Ошибка декодирования JSON: {json_str}")
        return None


def extract_key_info_from_paper(text, num_iterations=2):
    """
    Извлекает ключевую информацию по полям из текста статьи с итеративной проверкой качества.
    """
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
                continue
            
            check_msg = quality_check_prompt.format(
                extracted_json=json.dumps(extracted, indent=2)
            )
            
            response, msg_history = get_response_from_llm(
                msg=check_msg,
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
    """
    Обрабатывает все txt файлы в директории и сохраняет извлеченную информацию в JSON.
    """
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    for paper in papers:
        txt_path = Path(txt_dir) / f"{paper.arxiv_id}.txt"
        if not txt_path.exists():
            print(f"[ERROR] TXT для {paper.arxiv_id} не найден")
            continue

        output_path = output_dir_path / f"{paper.arxiv_id}.json"

        if output_path.exists() and output_path.stat().st_size > 0:
            print(f"[DEBUG] json для {paper.arxiv_id} уже существует, пропускаю извлечение")
            continue

        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read()

        print(f"[DEBUG] Обработка текста статьи {paper.arxiv_id}.txt...")
        info = extract_key_info_from_paper(text)

        if info:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(info, f, indent=2, ensure_ascii=False)
            print(f"[DEBUG] Ключевая информация для {paper.arxiv_id} сохранена в {output_path}")
        else:
            print(f"[ERROR] Не удалось извлечь информацию из {txt_path}")


def load_extracted_info(papers, extracted_dir='extracted_info'):
    """
    Загружает JSON файлы статей из extracted_info и возвращает список словарей.
    """
    papers_info = []
    
    for paper in papers:
        json_path = Path(extracted_dir) / f"{paper.arxiv_id}.json"
        if not json_path.exists():
            print(f"[ERROR] JSON для {paper.arxiv_id} не найден")
            continue

        with open(json_path, 'r', encoding='utf-8') as f:
            info = json.load(f)
            info['arxiv_id'] = paper.arxiv_id 
            papers_info.append(info)
    
    return papers_info


def group_papers_by_subtopics(papers_info, topic):
    """
    Группирует статьи по подтемам с помощью LLM.
    Возвращает словарь: {group_name: [paper_ids]}
    """
   
    papers_summary = []
    for paper in papers_info:
        summary = f"Title: {paper['title']}\nProblem: {paper['problem']}\nMethod: {paper['method']}\nResults: {paper['results']}"
        papers_summary.append(f"Paper {paper['arxiv_id']}:\n{summary}")
    
    papers_summary_text = "\n\n".join(papers_summary)
    print(f"[DEBUG] Сформирован текст для группировки статей по подтемам:\n{papers_summary_text}...")  
    
    response, _ = get_response_from_llm(
        msg=group_papers_by_subtopics_prompt.format(
            topic=topic, 
            papers_summary_text=papers_summary_text
            ),
        print_debug=False,
        rate_limit=OPENROUTER_API_EXTRACT_RATE_LIMIT_SEC
    )
    
    groups = extract_json_from_response(response)
    if not groups:
        print("[ERROR] Не удалось распарсить группы из ответа LLM")
        return {}
    
    return groups


def generate_group_analysis(papers_info, group_name, paper_ids, topic, num_iterations=3):
    """
    Генерирует анализ для одной группы статей с итеративным улучшением.
    """
    group_papers = [p for p in papers_info if p['arxiv_id'] in paper_ids]
    
    papers_text = []
    for paper in group_papers:
        text = f"Title: {paper['title']}\nProblem: {paper['problem']}\nMethod: {paper['method']}\nResults: {paper['results']}\nLimitations: {paper['limitations']}\nNovelty: {paper['novelty']}\nkey_findings: {paper['key_findings']}\nopen_questions: {paper['open_questions']}"
        papers_text.append(text)
    papers_combined = "\n\n".join(papers_text)
    
    analysis = None
    msg_history = None
    
    for iteration in range(num_iterations):
        if iteration == 0:
            # Первичная генерация анализа
            response, msg_history = get_response_from_llm(
                msg=group_analysis_prompt.format(
                    group_name=group_name, 
                    topic=topic, 
                    papers_combined=papers_combined
                ),
                print_debug=False,
                rate_limit=OPENROUTER_API_EXTRACT_RATE_LIMIT_SEC
            )
            analysis = response
        else:
            # Проверка качества и улучшение
            if analysis is None:
                continue
            
            check_response, msg_history = get_response_from_llm(
                msg=check_group_analysis_prompt.format(
                    group_name=group_name,
                    analysis=analysis,
                    papers_combined=papers_combined
                ),
                print_debug=False,
                msg_history=msg_history,
                rate_limit=OPENROUTER_API_EXTRACT_RATE_LIMIT_SEC
            )
            
            if "ПРИНЯТО" in check_response.upper():
                break
            else:
                # Генерируем улучшенную версию            
                response, msg_history = get_response_from_llm(
                    msg=improve_group_analysis_prompt.format(
                        group_name=group_name,
                        check_response=check_response,
                    ),
                    print_debug=False,
                    msg_history=msg_history,
                    rate_limit=OPENROUTER_API_EXTRACT_RATE_LIMIT_SEC
                )
                analysis = response
    
    return analysis

def generate_final_review(group_analyses, topic, num_iterations=3):
    """
    Объединяет анализы групп в итоговый обзор статьи с итеративным улучшением.
    """
    analyses_text = "\n\n".join([f"## {group_name}\n\n{analysis}" for group_name, analysis in group_analyses.items()])
    
    review = None
    msg_history = None
    
    for iteration in range(num_iterations):
        if iteration == 0:
            # Первичная генерация обзора
            response, msg_history = get_response_from_llm(
                msg=review_prompt.format(
                    topic=topic,
                    analyses_text=analyses_text
                ),
                print_debug=False,
                msg_history=msg_history,
                rate_limit=OPENROUTER_API_EXTRACT_RATE_LIMIT_SEC
            )
            review = response
        else:
            # Проверка качества и улучшение
            if review is None:
                continue
            
            check_response, msg_history = get_response_from_llm(
                msg=check_review_prompt.format(
                    topic=topic
                ),
                print_debug=False,
                msg_history=msg_history,
                rate_limit=OPENROUTER_API_EXTRACT_RATE_LIMIT_SEC
            )
            
            if "ПРИНЯТО" in check_response.upper():
                break
            else:
                # Генерируем улучшенную версию            
                response, msg_history = get_response_from_llm(
                    msg=improve_review_prompt.format(
                        topic=topic,
                        check_response=check_response,
                        ),
                    print_debug=False,
                    msg_history=msg_history,
                    rate_limit=OPENROUTER_API_EXTRACT_RATE_LIMIT_SEC
                )
                review = response
    
    return review


def perform_grouping_and_analysis(topic, papers, extracted_dir='extracted_info', output_dir='analysis_output'):
    """
    Выполняет полный этап 2: группировка и анализ.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Загружаем извлеченную информацию
    papers_info = load_extracted_info(papers=papers, extracted_dir=extracted_dir)
    print(f"[DEBUG] Загружено {len(papers_info)} статей из {extracted_dir}")
    
    # Группируем статьи
    groups = group_papers_by_subtopics(papers_info, topic)
    print(f"[DEBUG] Статьи сгруппированы в {len(groups)} групп: {groups}")
    input()
    
    # Сохраняем группы
    with open(output_path / 'groups.json', 'w', encoding='utf-8') as f:
        json.dump(groups, f, indent=2, ensure_ascii=False)
    
    # Генерируем анализ для каждой группы
    group_analyses = {}
    for group_name, paper_ids in groups.items():
        print(f"[DEBUG] Генерирую анализ для группы '{group_name}' ({len(paper_ids)} статей)")
        analysis = generate_group_analysis(papers_info, group_name, paper_ids, topic, num_iterations=3)
        group_analyses[group_name] = analysis
        
        # Сохраняем анализ группы
        safe_name = group_name.replace('/', '_').replace('\\', '_')
        with open(output_path / f'group_{safe_name}.txt', 'w', encoding='utf-8') as f:
            f.write(analysis)
    
    # Генерируем итоговый обзор
    print("[DEBUG] Генерирую итоговый обзор")
    final_review = generate_final_review(group_analyses, topic, num_iterations=3)
    
    # Сохраняем итоговый обзор
    with open(output_path / 'final_review.txt', 'w', encoding='utf-8') as f:
        f.write(final_review)
    
    print(f"[DEBUG] Анализ завершен. Результаты сохранены в {output_dir}")
    return final_review


# Промпт для извлечения информации из статьи
extraction_prompt = """
Тебе дан текст научной статьи. Проанализируй его и извлеки следующие поля в формате JSON:

- title: Название статьи
- problem: Главная проблема или исследовательский вопрос
- method: Методология или подход, использованный в работе
- results: Ключевые результаты или открытия
- limitations: Любые ограничения, упомянутые в статье
- novelty: Что новое или уникальное в этой работе
- key_findings: 1-2 предложения, резюмирующие основные результаты
- open_questions: Нерешенные вопросы или будущая работа, упомянутая в статье

Отвечай только объектом JSON, без дополнительного текста. Объект JSON выделяется таким блоком: ```json  ```

Текст статьи:
{text}
"""


# Промпт для проверки качества извлечения
quality_check_prompt = """
Оцени извлеченную информацию из статьи. Проверьте точность, полноту и согласованность с оригинальным текстом.

Извлеченная информация:
{extracted_json}

Дай обратную связь и предложи улучшения. Если информация удовлетворительна, напиши "ПРИНЯТО". В противном случае, предоставь исправленный JSON.
Отвечай либо "ПРИНЯТО", либо исправленным JSON.
"""


# Промпт для группировки статей по подтемам
group_papers_by_subtopics_prompt = """
Тебе дан список arxiv_id научных статей с извлеченной из текста статей ключевой информацией, связанной с темой: "{topic}".
Твоя задача - сгруппировать эти статьи по логическим подтемам. Каждая группа должна содержать статьи, которые адресуют похожие проблемы, методы или находки.

Инструкции:
1. Определи основные подтемы, основанные на содержании всех представленных статей. 
2. Всего подтем ДОЛЖНО БЫТЬ 4 ИЛИ МЕНЬШЕ. 
3. КАЖДАЯ СТАТЬЯ ДОЛЖНА БЫТЬ ОТНЕСЕНА РОВНО К ОДНОЙ ПОДТЕМЕ. НЕЛЬЗЯ ОСТАВЛЯТЬ СТАТЬЮ БЕЗ ОТНЕСЕНИЯ К КАКОЙ-ЛИБО ПОДТЕМЕ И НЕЛЬЗЯ ДУБЛИРОВАТЬ СТАТЬИ В НЕСКОЛЬКИХ ГРУППАХ.
4. Если статья одна в своей подтеме и не подходит ни под одну другую группу, то создай группу "Разное" и отнеси ее туда. Если все статьи можно логично сгруппировать, то не используй группу "Разное".
5. Для каждой подтемы перечисли arxiv_ids статей, которые к ней относятся.
6. Дай каждой подтеме описывающее ее название.

Отвечай только объектом JSON, где ключи - названия подтем, а значения - списки arxiv_ids статей.

Пример вывода:
{{
  "Subtopic 1": ["paper_id1", "paper_id2", "paper_id3"],
  "Subtopic 2": ["paper_id4", "paper_id5"],
  "Miscellaneous": ["paper_id6"]
}}

Извлеченная ключевая информация по статьям:
{papers_summary_text}

Отвечай только объектом JSON.
"""

# Промпт для генерации анализа группы статей
group_analysis_prompt = """
Тебе дана группа научных статей по подтеме "{group_name}" в рамках более широкой темы "{topic}".

Статьи в этой группе:
{papers_combined}

Твоя задача - предоставить комплексный анализ этой подтемы на основе статей. Включи:
1. **Обзор**: Быстрый обзор того, что охватывает эта подтема.
2. **Основные проблемы**: Распространенные проблемы, рассматриваемые в этих статьях.
3. **Используемые методы**: Обзор методологий, применяемых в исследованиях.
4. **Основные результаты**: Синтез ключевых результатов и открытий.
5. **Ограничения**: Распространенные ограничения среди статей.
6. **Новые вклады**: Какие новые идеи или подходы представлены.
7. **Открытые вопросы**: Нерешенные проблемы и направления будущих исследований.
8. **Сравнение**: Как эти статьи сравниваются/контрастируют друг с другом.

Пиши анализ в структурированном формате с четкими заголовками.
"""


# Промпт для проверки качества анализа группы статей
check_group_analysis_prompt = """
Проверь анализ подтемы "{group_name}". Проверь на полноту, точность, согласованность, relevance к статьям, научную строгость и ясность структуры.

Анализ:
{analysis}

Резюме статей:
{papers_combined}

Дай подробную обратную связь по любым слабым местам, несоответствиям или областям для улучшения. Будь строг: укажи отсутствующую информацию, логические пробелы, неясные объяснения или недостаточную глубину. Если анализ удовлетворительный, скажи "ПРИНЯТО". В противном случае, предоставь конкретные, действенные предложения для улучшения.
Отвечай либо "ПРИНЯТО", либо подробными предложениями для улучшения.
"""


# Промпт для улучшения анализа группы статей
improve_group_analysis_prompt = """
Основываясь на обратной связи написанного тобой анализа, улучши анализ подтемы "{group_name}".

Фидбек на текущий анализ:
{check_response}

Предоставь улучшенную версию анализа, учитывая обратную связь.
"""


# Промпт для генерации итогового обзора
review_prompt = """
Тебе даны анализы различных подтем в рамках темы "{topic}". Твоя задача - синтезировать их в комплексный обзор литературы.

Анализы подтем:
{analyses_text}

Структурируй финальную статью обзора с следующими разделами:
1. **Введение**: Обзор темы и ее важности.
2. **Фон и связанные работы**: Общий контекст из подтем.
3. **Методы и подходы**: Общие методологии среди подтем.
4. **Ключевые находки и результаты**: Синтез главных результатов.
5. **Ограничения и вызовы**: Распространенные ограничения.
6. **Будущие направления**: Открытые вопросы и возможности исследований.
7. **Заключение**: Итоги и применения.

Убедись, что обзор построен логично и связывает подтемы согласованно. Сделай его подходящим для публикации как обзорную научную статью.
"""


# Промпт для проверки качества итогового обзора
check_review_prompt = """
Проверь написанную тобой обзорную статью по теме "{topic}". Проверь на полноту, согласованность, логический поток, научную строгость, подходящесть для публикации, глубину анализа и интеграцию подтем.

Дай детальную обратную связь по любым слабым местам, несоответствиям или областям для улучшения. Будь строг: укажи отсутствующую информацию, логические пробелы, неясные объяснения или недостаточную глубину. Если анализ удовлетворительный, скажи "ПРИНЯТО". В противном случае, предоставь конкретные, действенные предложения для улучшения.
"""


# Промпт для улучшения итогового обзора
improve_review_prompt = """
Основываясь на обратной связи, улучши обзорную статью по теме "{topic}".

Фидбек:
{check_response}

Приведи улучшенную версию обзора, учитывая обратную связь.
"""


if __name__ == "__main__":
    txt_dir = "txts"
    output_dir = "extracted_info"
    extract_key_info_from_papers([ArxivPaper(arxiv_id="2409.11363v1", title="Test Paper", abstract="Test abstract", authors=[],published=None, updated=None, doi=None, pdf_url=None, source_url=None)], output_dir=output_dir, txt_dir=txt_dir)
