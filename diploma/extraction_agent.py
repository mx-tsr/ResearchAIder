import json
import os
import re

from pathlib import Path

from arxiv_agent import ArxivPaper
from llm_agent import get_response_from_llm

OPENROUTER_API_EXTRACT_RATE_LIMIT_SEC = 7.0 

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
            
            check_msg = extraction_quality_check_prompt.format(
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
    Группирует статьи по подтемам с помощью LLM с refinement.
    Возвращает словарь: {group_name: [paper_ids]}
    """
   
    papers_summary = []
    for paper in papers_info:
        summary = f"Title: {paper['title']}\nProblem: {paper['problem']}\nCompared baselines: {paper['compared_baselines']}\nResults: {paper['results']}"
        papers_summary.append(f"Paper {paper['arxiv_id']}:\n{summary}")
    
    papers_summary_text = "\n\n".join(papers_summary)
    print(f"[DEBUG] Сформирован текст для группировки статей по подтемам:\n{papers_summary_text}\n")  
    
    groups = None
    for iteration in range(2):  # 2 итерации refinement
        if iteration == 0:
            # Первичная группировка
            response, _ = get_response_from_llm(
                msg=group_papers_by_subtopics_prompt.format(
                    topic=topic, 
                    papers_summary_text=papers_summary_text
                    ),
                print_debug=False,
                temperature=0.1,
                rate_limit=OPENROUTER_API_EXTRACT_RATE_LIMIT_SEC
            )
            groups = extract_json_from_response(response)
            print(f"[DEBUG] Результат первичной группировки: {groups}\n")
        else:
            # Проверка и улучшение
            if groups is None:
                continue
            
            # Проверяем распределение
            all_paper_ids = set()
            duplicates = []
            for group_papers in groups.values():
                for paper_id in group_papers:
                    if paper_id in all_paper_ids:
                        duplicates.append(paper_id)
                    all_paper_ids.add(paper_id)
            
            expected_papers = {p['arxiv_id'] for p in papers_info}
            missing = expected_papers - all_paper_ids
            extra = all_paper_ids - expected_papers
            
            if not missing and not duplicates and not extra:
                break  # Все хорошо
            
            # Фидбек для улучшения
            feedback = ""
            if missing:
                feedback += f"Не распределены статьи: {list(missing)}. "
            if duplicates:
                feedback += f"Дубликаты статей: {duplicates}. "
            if extra:
                feedback += f"Лишние статьи: {list(extra)}. "
            
            print(f'[DEBUG] Фидбек для улучшения групп: {feedback}')
            
            improve_prompt = f"""
Текущая группировка:
{json.dumps(groups, indent=2)}

Проблемы: {feedback}

Исправь группировку: исправь все перечисленные проблемы. Каждая статья должна быть распределена ровно в одну самую подходящую группу без дубликатов. 
Ни одна статья не должна остаться вне группы. Суммарно групп должно быть 4 или меньше. Если групп получилось 5, то их число нужно сократить. Если статья не определна ни в одну группу, нужно определить ее в самую подходящую. 

Ответь только исправленным JSON.
"""
            
            response, _ = get_response_from_llm(
                msg=improve_prompt,
                print_debug=False,
                temperature=0.1,
                rate_limit=OPENROUTER_API_EXTRACT_RATE_LIMIT_SEC
            )
            groups = extract_json_from_response(response)
            print(f"[DEBUG] Результат повторной группировки: {groups}\n")
    
    if not groups:
        print("[ERROR] Не удалось распарсить группы из ответа LLM")
        return {}
    
    return groups


def generate_group_analysis(papers_info, group_name, paper_ids, topic, num_iterations=3):
    """
    Генерирует анализ для одной группы статей с итеративным улучшением через объединённый промпт.
    """
    group_papers = [p for p in papers_info if p['arxiv_id'] in paper_ids]
    
    papers_text = []
    for paper in group_papers:
        text = f"Title: {paper['title']}\nProblem: {paper['problem']}\nCompared baselines: {paper['compared_baselines']}\nResults: {paper['results']}\nLimitations: {paper['limitations']}\nNovelty: {paper['novelty']}\nkey_findings: {paper['key_findings']}\nopen_questions: {paper['open_questions']}"
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
                msg_history=msg_history,
                rate_limit=OPENROUTER_API_EXTRACT_RATE_LIMIT_SEC
            )
            analysis = response
        else:
            # Итеративная проверка и улучшение с объединённым промптом
            if analysis is None:
                continue
            
            reflection_prompt = group_analysis_reflection_prompt.format(group_name=group_name)
            reflection_prompt += f"\n\nТекущий анализ:\n{analysis}\n\nРезюме статей:\n{papers_combined}"
            
            response, msg_history = get_response_from_llm(
                msg=reflection_prompt,
                print_debug=False,
                msg_history=msg_history,
                rate_limit=OPENROUTER_API_EXTRACT_RATE_LIMIT_SEC
            )
            
            # Извлекаем текст из ответа
            if "I am done" in response:
                # Извлекаем анализ из блока RESPONSE
                match = re.search(r'RESPONSE:\s*```\n(.*?)\n```', response, re.DOTALL)
                if match:
                    analysis = match.group(1)
                break
            else:
                # Извлекаем улучшенный анализ из блока RESPONSE
                match = re.search(r'RESPONSE:\s*```\n(.*?)\n```', response, re.DOTALL)
                if match:
                    analysis = match.group(1)
    
    return analysis

def extract_relevant_parts(analyses, relevant_headers):
    """
    Извлекает релевантные части из анализов по заголовкам (маркеры ##).
    Использует структурированный формат с явными маркерами разделов.
    """
    extracted = {}
    
    for group_name, analysis in analyses.items():
        group_sections = {}
        lines = analysis.split('\n')
        current_section = None
        current_content = []
        
        for line in lines:
            # Проверяем, является ли строка заголовком раздела (начинается с ##)
            if line.startswith('##'):
                # Сохраняем предыдущий раздел
                if current_section:
                    group_sections[current_section] = '\n'.join(current_content).strip()
                
                # Извлекаем название раздела
                current_section = line.replace('##', '').strip()
                current_content = []
            elif current_section:
                # Добавляем содержимое к текущему разделу
                current_content.append(line)
        
        # Сохраняем последний раздел
        if current_section:
            group_sections[current_section] = '\n'.join(current_content).strip()
        
        extracted[group_name] = group_sections
    
    # Извлекаем нужные разделы из каждой группы
    result_text = []
    for header in relevant_headers:
        for group_name, sections in extracted.items():
            for section_name, content in sections.items():
                if header.lower() in section_name.lower():
                    if content:
                        result_text.append(f"Из группы '{group_name}' ({section_name}):\n{content}")
    
    return '\n\n'.join(result_text) if result_text else f"[Не найдено релевантных разделов для: {relevant_headers}]"


def generate_final_review(group_analyses, topic, num_iterations=3):
    """
    Генерирует итоговый обзор статьи по фиксированным разделам с объединённым промптом.
    БЕЗ ссылок на статьи - они добавляются отдельно.
    """
    # Фиксированные разделы финального обзора
    sections = [
        "Введение",
        "Предметная область", 
        "Методы и Подходы",
        "Ключевые Результаты и Открытия",
        "Ограничения и Проблемы",
        "Будущие Направления",
        "Заключение"
    ]
    
    # Маппинг разделов к частям анализов групп
    section_mapping = {
        "Введение": ["Обзор подтемы"],
        "Предметная область": ["Обзор подтемы", "Основные идеи и методы"],
        "Методы и Подходы": ["Основные идеи и методы", "Сильные стороны и ограничения"],
        "Ключевые Результаты и Открытия": ["Совпадающие выводы и подтверждающие результаты", "Различия и противоречия"],
        "Ограничения и Проблемы": ["Сильные стороны и ограничения", "Различия и противоречия"],
        "Будущие Направления": ["Пробелы исследований и открытые вопросы"],
        "Заключение": ["Ключевые выводы для итогового мета-анализа"]
    }
    
    # Генерировать каждый раздел
    final_sections = {}
    
    for section_name in sections:
        print(f"[DEBUG] Генерирую раздел: {section_name}")
        relevant_parts = extract_relevant_parts(group_analyses, section_mapping[section_name])
        
        section_content = None
        msg_history = None
        
        for iteration in range(num_iterations):
            if iteration == 0:
                # Первичная генерация раздела БЕЗ ссылок
                prompt = f"""
Напиши раздел "{section_name}" научной обзорной статьи по теме "{topic}".

Используй эту релевантную информацию из анализов групп:
{relevant_parts}

Требования:
- Научный стиль, синтез (не пересказ).
- Сравнивай подходы между группами.
- НЕ добавляй ссылки на статьи - они будут добавлены отдельно.
- Учитывай противоречия и консенсус.
- Раздел уровня обзорной публикации.
- Пиши ТОЛЬКО ТОТ РАЗДЕЛ, БЕЗ ВЫВОДОВ К НЕМУ.
"""
                response, msg_history = get_response_from_llm(
                    msg=prompt,
                    print_debug=False,
                    msg_history=msg_history,
                    rate_limit=OPENROUTER_API_EXTRACT_RATE_LIMIT_SEC
                )
                section_content = response
            else:
                # Итеративная проверка и улучшение
                if section_content is None:
                    continue
                
                reflection_prompt = review_section_reflection_prompt.format(
                    section_name=section_name,
                    topic=topic
                )
                reflection_prompt += f"\n\nТекущий раздел:\n{section_content}\n\nРелевантная информация:\n{relevant_parts}"
                
                response, msg_history = get_response_from_llm(
                    msg=reflection_prompt,
                    print_debug=False,
                    msg_history=msg_history,
                    rate_limit=OPENROUTER_API_EXTRACT_RATE_LIMIT_SEC
                )
                
                # Извлекаем текст из ответа
                if "I am done" in response:
                    match = re.search(r'RESPONSE:\s*```\n(.*?)\n```', response, re.DOTALL)
                    if match:
                        section_content = match.group(1)
                    break
                else:
                    match = re.search(r'RESPONSE:\s*```\n(.*?)\n```', response, re.DOTALL)
                    if match:
                        section_content = match.group(1)
        
        final_sections[section_name] = section_content
    
    # Собрать итоговую статью БЕЗ ссылок
    full_review = "\n\n".join([f"# {section_name}\n\n{content}" for section_name, content in final_sections.items()])
    
    print(f'[DEBUG] Полный обзор БЕЗ ССЫЛОК сформирован: {full_review}\n')
    
    # Финальная проверка всей статьи
    msg_history = None
    for check_iteration in range(3):
        check_prompt = final_review_check_prompt.format(topic=topic)
        check_prompt += f"\n\nСтатья:\n{full_review}"
        
        response, msg_history = get_response_from_llm(
            msg=check_prompt,
            print_debug=False,
            msg_history=msg_history,
            rate_limit=OPENROUTER_API_EXTRACT_RATE_LIMIT_SEC
        )
        
        if "I am done" in response:
            match = re.search(r'RESPONSE:\s*```\n(.*?)\n```', response, re.DOTALL)
            if match:
                full_review = match.group(1)
            break
        else:
            match = re.search(r'RESPONSE:\s*```\n(.*?)\n```', response, re.DOTALL)
            if match:
                full_review = match.group(1)
    
    return full_review  


def extract_response_block(response, label='RESPONSE'):
    """Извлекает содержимое блока RESPONSE из ответа LLM."""
    match = re.search(rf'{label}:\s*```\n(.*?)\n```', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    print('f[ERROR] Не найден блок {label} в ответе\n')
    return None


def add_citations_to_review(review_text, papers_info, topic, num_iterations=3):
    """
    Добавляет ссылки на статьи в готовый обзор (отдельный этап).
    Использует msg_history для контекста.
    """
    papers_list = []
    for p in papers_info:
        papers_list.append(f"- arxiv_id: {p['arxiv_id']}\n  title: {p['title']}\n  problem: {p['problem']}\n results: {p['results']}")
    papers_info_str = "\n".join(papers_list)

    review_with_citations = review_text
    msg_history = None

    for iteration in range(num_iterations):
        if iteration == 0:
            add_citations_msg = add_citations_prompt.format(
                topic=topic,
                review_text=review_with_citations,
                papers_info=papers_info_str
            )

            response, msg_history = get_response_from_llm(
                msg=add_citations_msg,
                print_debug=False,
                msg_history=msg_history,
                rate_limit=OPENROUTER_API_EXTRACT_RATE_LIMIT_SEC
            )

            parsed = extract_response_block(response, label='RESPONSE')
            if parsed:
                # Если модель вернула обрезанный текст или не сохранила структуру, оставляем прежний обзор
                if all(header in parsed for header in ["# Введение", "# Предметная область", "# Методы и Подходы", "# Ключевые Результаты и Открытия", "# Ограничения и Проблемы", "# Будущие Направления", "# Заключение"]):
                    review_with_citations = parsed
                else:
                    print("[DEBUG] Добавление цитат вернуло некорректный текст, сохранён предыдущий обзор без изменений.")
                    break

        else:
            check_msg = f"""
Проверь этот обзор на наличие всех необходимых ссылок. Не меняй текст, кроме добавления ссылок. Если ссылки добавлены полностью, ответь в RESPONSE точно тем же текстом обзора и добавь в мыслях 'I am done'. Если нужно добавить ещё ссылки, верни обновлённый обзор и НЕ пиши 'I am done'.

THOUGHT:

RESPONSE:
```
{review_with_citations}
```
"""
            check_response, msg_history = get_response_from_llm(
                msg=check_msg,
                print_debug=False,
                msg_history=msg_history,
                rate_limit=OPENROUTER_API_EXTRACT_RATE_LIMIT_SEC
            )
            check_parsed = extract_response_block(check_response, label='RESPONSE')
            if "I am done" in check_response and check_parsed == review_with_citations:
                break
            if check_parsed and len(check_parsed) > 0:
                review_with_citations = check_parsed

    return review_with_citations


def generate_hypotheses_section(final_review, papers_info, topic, num_iterations=2):
    """
    Генерирует раздел гипотез на основе имеющихся анализов и использованных статей.
    """
    paper_summaries = []
    for p in papers_info:
        paper_summaries.append(
            f"- arxiv_id: {p['arxiv_id']}\n  title: {p['title']}\n  problem: {p.get('problem', '')}\n  results: {p.get('results', '')}\n  open_questions: {p.get('open_questions', '')}\n implicit_gaps: {p.get('implicit_gaps', '')}"
        )
    papers_info_str = "\n".join(paper_summaries)

    prompt = hypotheses_prompt.format(
        topic=topic,
        final_review=final_review,
        papers_info=papers_info_str,
    )
    
    response, _ = get_response_from_llm(
        msg=prompt,
        print_debug=False,
        rate_limit=OPENROUTER_API_EXTRACT_RATE_LIMIT_SEC,
    )
    hypotheses_text = extract_response_block(response, label='RESPONSE')
    print(f"[DEBUG] Изначальный текст раздела гипотезы: {hypotheses_text}\n")

    for _ in range(num_iterations):
        reflection_msg = hypotheses_reflection_prompt.format(
            topic=topic,
            hypotheses=hypotheses_text
        )
        response, _ = get_response_from_llm(
            msg=reflection_msg,
            print_debug=False,
            rate_limit=OPENROUTER_API_EXTRACT_RATE_LIMIT_SEC,
        )
        if "I am done" in response:
            improved = extract_response_block(response, label='RESPONSE')
            if improved:
                hypotheses_text = improved
            break
        improved = extract_response_block(response, label='RESPONSE')
        print(f"[DEBUG] Улучшенный текст гипотезы: {improved}\n")

        if improved:
            hypotheses_text = improved

    return hypotheses_text


def perform_grouping_and_analysis(topic, papers, extracted_dir='extracted_info', output_dir='analysis_output'):
    """
    Выполняет полный этап 2: группировка, анализ и генерация обзора.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Загружаем извлеченную информацию
    papers_info = load_extracted_info(papers=papers, extracted_dir=extracted_dir)
    print(f"[DEBUG] Загружено {len(papers_info)} статей из {extracted_dir}")
    
    # Группируем статьи
    print(f"[DEBUG] Начинаю группировку статей на группы")
    groups = group_papers_by_subtopics(papers_info, topic)
    print(f"[DEBUG] Статьи сгруппированы в {len(groups)} групп: {groups}")
    
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
    
    # Генерируем итоговый обзор БЕЗ ссылок
    print("[DEBUG] Генерирую итоговый обзор (БЕЗ ССЫЛОК)")
    final_review = generate_final_review(group_analyses, topic, num_iterations=3)
    
    # Сохраняем обзор без ссылок
    with open(output_path / 'final_review_no_citations.txt', 'w', encoding='utf-8') as f:
        f.write(final_review)
    
    # Генерируем раздел гипотез и дальнейших исследований на основе пробелов и 10 статей
    print("[DEBUG] Генерирую раздел гипотез и дальнейших исследований")
    hypotheses_text = generate_hypotheses_section(final_review, papers_info, topic, num_iterations=2)
    final_review = final_review.strip() + "\n\n# Гипотезы и дальнейшие исследования\n\n" + hypotheses_text

    # Сохраняем обзор без ссылок, уже с разделом гипотез
    with open(output_path / 'final_review_no_citations.txt', 'w', encoding='utf-8') as f:
        f.write(final_review)

    # Добавляем ссылки на статьи
    print("[DEBUG] Добавляю ссылки на статьи в обзор")
    final_review_with_citations = add_citations_to_review(final_review, papers_info, topic, num_iterations=3)
    
    # Сохраняем итоговый обзор со ссылками
    with open(output_path / 'final_review.txt', 'w', encoding='utf-8') as f:
        f.write(final_review_with_citations)
    
    print(f"[DEBUG] Анализ завершен. Результаты сохранены в {output_dir}")
    return final_review_with_citations


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
{text}
"""


# Промпт для проверки качества извлечения
extraction_quality_check_prompt = """
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

Отнеси каждую статью ТОЛЬКО К ОДНОЙ ПОДТЕМЕ. Отвечай только объектом JSON.
"""

# Промпт для генерации анализа группы статей
group_analysis_prompt = """
Ты — агент научного синтеза, создающий промежуточный аналитический артефакт для последующей генерации обзорной научной статьи по теме "{topic}".

Тебе дана группа научных статей "{group_name}", объединенных одной подтемой.

ВАЖНО: не пересказывай статьи по отдельности. Нужно провести сравнительный мета-анализ группы работ.

Сформируй подробный анализ со СЛЕДУЮЩЕЙ СТРУКТУРОЙ разделов (обязательно выделяй каждый раздел строкой начинающейся с ##):

## Обзор подтемы
Какую исследовательскую область покрывают работы? Почему направление важно?

## Основные идеи и методы
Какие подходы предлагаются? Сгруппируй методы по типам. Сравнивай подходы между собой.

## Совпадающие выводы и подтверждающие результаты
Какие идеи подтверждаются несколькими работами? Указывай какие статьи это подтверждают.

## Различия и противоречия
Где статьи расходятся? Чем объясняются различия?

## Сильные стороны и ограничения
Сравни преимущества и недостатки подходов. Отрази ограничения, упомянутые в работах.

## Данные и экспериментальные оценки
Какие датасеты, метрики и способы оценки используются? Что общего и что различается?

## Пробелы исследований и открытые вопросы
Что остается нерешенным? Какие направления изучены недостаточно?

## Ключевые выводы для итогового мета-анализа
Какие выводы важно сохранить для дальнейшего построения обзорной статьи?

ТРЕБОВАНИЯ:
- Анализ должен быть подробным и вдумчивым.
- Это промежуточный материал для финальной обзорной статьи.
- Не делай утверждений без опоры на статьи.
- Делай акцент на сравнении, синтезе и связях.

Статьи в этой группе:
{papers_combined}

Пиши анализ в структурированном формате с четкими заголовками, начиная каждый раздел с ##.
"""


# Объединённый промпт для проверки и улучшения анализа группы статей
group_analysis_reflection_prompt = """
Проверь анализ подтемы "{group_name}" на полноту, точность, согласованность, relevance к статьям, научную строгость и ясность структуры.
Проверь:
1. Все ли статьи действительно отражены в каждом разделе.
2. Достаточно ли глубок анализ.
3. Есть ли сравнение, противоречия и связи между работами.
4. Выявлены ли пробелы исследований.
5. Достаточно ли качественный это промежуточный артефакт для итогового обзора.

Отвечай в формате:

THOUGHT:
<твои размышления об анализе>

RESPONSE:
```
<обновленный анализ или старый текст статьи, если текст удовлетворил>
```

Если анализ удовлетворительный и не нужны изменения, включи I am done в THOUGHT, но в RESPONSE верни ТОЧНО ТАКОЙ ЖЕ предыдущий текст анализа.
ВКЛЮЧАЙ I am done ТОЛЬКО ЕСЛИ ТЫ НЕ ДЕЛАЕШЬ ИЗМЕНЕНИЙ.
"""


# Объединённый промпт для написания и итеративного улучшения раздела обзора
review_section_reflection_prompt = """
СТРОГО СОБЛЮДАЙ ФОРМАТ ОТВЕТА:

THOUGHT:
<твои размышления о разделе>

RESPONSE:
```
<улучшенный раздел или старый текст раздела, если текст удовлетворил>
```

Проверь раздел "{section_name}" обзорной статьи по теме "{topic}".

Оцени:
- полноту информации из предоставленных материалов
- согласованность и логичность изложения
- научную строгость
- глубину обработки материала
- не потеряна ли важная информация из анализов групп

Если раздел тебя удовлетворил, ОБЯЗАТЕЛЬНО напиши "I am done" в разделе THOUGHT и верни ТОЧНО ТАКОЙ ЖЕ предыдущий текст раздела в RESPONSE.
ВКЛЮЧАЙ "I am done" ТОЛЬКО ЕСЛИ ТЫ НЕ ДЕЛАЕШЬ ИЗМЕНЕНИЙ.

ВАЖНО: Пиши в этом промпте ТОЛЬКО ТОТ РАЗДЕЛ, БЕЗ ВЫВОДОВ И ЗАКЛЮЧЕНИЙ К НЕМУ.
"""

# Промпт для проверки всего обзора (после написания без ссылок)
final_review_check_prompt = """
СТРОГО СОБЛЮДАЙ ФОРМАТ ОТВЕТА:

THOUGHT:
<твои размышления о статье>

RESPONSE:
```
<улучшенная статья или старый текст статьи, если текст удовлетворил>
```

Проверь целиком обзорную статью по теме "{topic}" на:
- согласованность разделов
- повторы
- пропущенные переходы
- противоречия
- недостающий синтез
- полноту раздела с пробелами исследований

Если статья тебя удовлетворяет, ОБЯЗАТЕЛЬНО включи "I am done" в мыслях и верни ТОЧНО ТАКОЙ ЖЕ предыдущий текст статьи.
ВКЛЮЧАЙ "I am done" ТОЛЬКО ЕСЛИ ТЫ НЕ ДЕЛАЕШЬ ИЗМЕНЕНИЙ.
"""

# Промпт для добавления ссылок на статьи в обзор
add_citations_prompt = """
Ты — агент, отвечающий за добавление ссылок на статьи в готовый обзор.

У тебя есть обзорная статья по теме "{topic}" без ссылок [CITATION: ...], и набор статей, которые использовались для его написания.
Твоя задача: добавить ссылки только в те места, где они действительно нужны для подтверждения важных утверждений или прямых отсылок к работам.

ВАЖНО:
- Не меняй текст обзора больше, чем необходимо.
- Не переставляй строки и не переписывай фразы.
- Не добавляй новые идеи, не удаляй ничего.
- Добавляй ссылки только после значимых утверждений или сравнений.
- Ссылки должны быть в формате: [CITATION: arxiv_id | название статьи]

Отвечай строго в формате:

THOUGHT:
<кратко, какие ссылки ты добавил и почему>

RESPONSE:
```
<полный текст обзора с добавленными ссылками>
```

Текст обзора БЕЗ ССЫЛОК:
{review_text}

Доступные статьи (используй ТОЧНЫЕ arxiv_id и названия):
{papers_info}
"""

# Промпт для генерации раздела гипотез и дальнейших исследований
hypotheses_prompt = """
Ты — амбициозный научный исследователь, который анализирует итоговый обзор по теме "{topic}" и 10 использованных статей и хочет выдвинуть новые, новаторские, актуальные гипотезы, сформулированные в виде названий научных статей, дальнейшее исследование которых внесло бы значительный вклад в область "{topic}".

Твоя задача: на основе итоговой обзорной статьи и ключевых элементов статей, по которым писался итоговый обзор, сформулировать 2-3 конкретные и новаторские исследовательские гипотезы в виде названий научных статей, по которым можно провести дальнейшее исследование. Все гипотезы записываются в виде раздела "Гипотезы для дальнейшего исследлования", который затем будет добавлен к итоговому обзору всей темы. 

Требования:
- Гипотезы должны быть достаточно узкими и практичными и сформулированы в виде названия статей.
- Гипотеза НЕ должна существенно пересекаться с существующей литературой или быть уже хорошо изученной.
- В разделе ты должен объяснить, почему каждая гипотеза является новой и не покрыта текущей литературой.
- Укажи, какие пробелы в литературе поддерживают каждую гипотезу.
- Не генерируй шаблонные идеи, фокусируйся на слепых зонах, выявленных из предоставленных статей.
- Отвечай в формате текста, готового для раздела обзора.
- Отвечай ТОЛЬКО на русском.

Отвечай СТРООГО в формате:

THOUGHT:
<кратко, почему ты выбрал именно эти гипотезы>

RESPONSE:
```
<текст раздела "Гипотезы для дальнейшего исследования">
```

Текст итогового обзора: 
{final_review}

Информация о каждой из 10 использованных статей:
{papers_info}
"""

hypotheses_reflection_prompt = """
У тебя есть написанный раздел "Гипотезы для дальнейшего исследования", в которых описаны гипотезы и идеи в виде названий научных статей с пояснениями, почему они могут быть новаторскими и актуальными. 
Ты хочешь проверить, новы эти гипотезы или нет. То есть, не пересекаются ли они существенно с существующей литературой или уже хорошо изучены.
Будь строгим критиком новизны, убедись, что в этой гипотезе или идее содержится достаточный вклад в область исследования.

THOUGHT:
<размышления о том, как улучшить гипотезы и обоснование>

RESPONSE:
```
<улучшенный текст раздела "Гипотезы для дальнейшего исследования">
```

Текст раздела "Гипотезы для дальнейшего исследования": 
{hypotheses}

Если после анализа новизны ты решил, что раздел написан хорошо, не требует изменений, а гипотезы новы и актуальны, то включи "I am done" в THOUGHT и верни ТОЧНО ТАКОЙ ЖЕ текст раздела в RESPONSE.  
"""


if __name__ == "__main__":
    group_analyses = {
        'Data Science Automation': '''## Обзор подтемы
Данная исследовательская область фокусируется на автоматизации процессов науки о данных (Data Science Automation) с использованием многоагентных систем (Multi-Agent Systems, MAS) и больших языковых моделей (LLM). Основная цель — преодоление ограничений существующих методов, которые страдают от хрупкости (brittleness), недостатка воспроизводимости, интерпретируемости и устойчивости (robustness) при работе с растущим объемом, сложностью и гетерогенностью данных.

Важность направления обусловлена тем, что современные задачи науки о данных требуют не просто достижения точного результата, но и обеспечения надежности, возможности объяснения решений и способности адаптироваться к сложным, неструктурированным данным. Многоагентные системы, управляемые LLM, предлагают архитектуру для выполнения сложных, многоэтапных, исследовательских задач, требующих планирования, адаптации и исправления ошибок.

## Основные идеи и методы
Представленные работы предлагают три основных подхода к автоматизации науки о данных, которые можно сгруппировать по их основному фокусу: стабильность, планирование и архитектура агента.

1. Подход, основанный на стабильности и воспроизводимости (VDSAgents):
    *   Метод: PCS-Guided (Principles of Science-Guided) подход.
    *   Суть: Декомпозиция всего конвейера науки о данных на последовательные этапы рассуждений, использование модульных функций (mltools) и оценка чувствительности данных через пертурбированные варианты.
    *   Цель: Улучшение стабильности, интерпретируемости и воспроизводимости процесса.

2. Подход, основанный на планировании и ансамблевой оптимизации (SPIO):
    *   Метод: Многопутевое планирование (Multi-path planning) и ансамблевая оптимизация (Ensemble optimization).
    *   Суть: Использование LLM для генерации множества стратегий (путей) на каждом этапе и последующее сравнение их с помощью ансамблевых методов (например, Top-k=2) для выбора наиболее надежного плана.
    *   Цель: Повышение точности и робастности за счет исследования разнообразных стратегий.

3. Подход, основанный на архитектуре агента и адаптивности (DatawiseAgent):
    *   Метод: Нейросетевая архитектура, ориентированная на блокноты (Notebook-Centric LLM Agent Framework) с FST-based (Finite State Transition) многостадийной архитектурой.
    *   Суть: Создание гибкого фреймворка, включающего модули для планирования и исправления кода (code repair), а также унифицированное представление взаимодействия.
    *   Цель: Обеспечение адаптивности, способности к долгосрочному планированию и робастному восстановлению после сбоев.

Сравнение подходов:
   VDSAgents фокусируется на качестве и стабильности* процесса, обеспечивая воспроизводимость через структурированное рассуждение.
   SPIO фокусируется на эффективности и надежности* решения, исследуя пространство возможных стратегий.
   DatawiseAgent фокусируется на архитектуре и способности* агента к адаптации и самокоррекции в динамической среде.

## Совпадающие выводы и подтверждающие результаты
Несмотря на различия в методологиях, все три работы сходятся в следующем:

1. Улучшение производительности и робастности: Все системы демонстрируют способность превосходить существующие базовые системы (AutoKaggle, DataInterpreter) и улучшать точность прогнозирования. SPIO улучшает точность на 5.6% по сравнению с лучшими системами, а VDSAgents показывает превосходную стабильность и устойчивость, особенно на сложных и зашумленных наборах данных.
2. Важность многоэтапного планирования: Успех в автоматизации зависит от способности системы не просто выполнять команды, а проводить последовательное планирование и исследование стратегий. SPIO и DatawiseAgent подчеркивают необходимость многоступенчатой архитектуры для решения сложных задач.
3. Роль модульности и восстановления ошибок: Для достижения высокой надежности необходимы механизмы, позволяющие агентам адаптироваться к ошибкам. DatawiseAgent явно демонстрирует, что модули планирования и исправления кода критически важны для успешной автоматизации.

## Различия и противоречия
Основные различия между работами лежат в акценте: стабильность против планирования против архитектуры.

1. Фокус на стабильности vs. Фокус на планировании:
       VDSAgents акцентирует внимание на процессе (PCS) и воспроизводимости как ключевых элементах. Это более методологический подход, направленный на то, как* процесс должен быть структурирован.
       SPIO акцентирует внимание на результате (точность) и стратегии (ансамблирование) как способе достижения оптимального решения. Это более стратегический подход, направленный на то, какой* путь выбрать.

2. Роль LLM и архитектура:
       DatawiseAgent предлагает более детализированную архитектуру* (FST-based multi-stage architecture), которая явно интегрирует планирование и восстановление ошибок в единый фреймворк.
       VDSAgents использует LLM в контексте  guided* процесса, где планирование (PCS) служит основой для оценки стабильности.
       SPIO использует LLM преимущественно как генератор* стратегий, где планирование является механизмом для генерации множества вариантов.

3. Ограничения: SPIO признает, что многопутевое планирование не гарантирует глобально оптимальных конвейеров, тогда как VDSAgents фокусируется на улучшении стабильности и интерпретируемости, что является отдельным, но взаимосвязанным вызовом.

## Сильные стороны и ограничения
| Работа | Сильные стороны | Ограничения |
| :--- | :--- | :--- |
| VDSAgents | Достигает превосходной стабильности и робастности; использует методологически обоснованный подход (PCS) для обеспечения воспроизводимости; эффективно оценивает чувствительность данных. | Не явно оговорены ограничения по масштабируемости или обобщению за пределы конкретных доменов. |
| SPIO | Эффективно использует многопутевое планирование и ансамблевое обучение для повышения точности; демонстрирует способность исследовать разнообразные стратегии. | Не гарантирует глобально оптимальный конвейер; высокая чувствительность к параметрам LLM, выбираемым для генерации путей. |
| Datawise | Предлагает комплексную архитектуру для обработки непредсказуемых данных через модули планирования и выполнения, что обеспечивает высокую адаптивность. | Требует сложной архитектуры для управления многоступенчатым процессом планирования и выполнения. |

## Заключение

Три подхода (VDSA, SPIO, Datawise) демонстрируют различные, но взаимодополняющие стратегии для решения проблемы надежного и адаптивного выполнения задач в области машинного обучения. VDSA фокусируется на стабильности, SPIO — на оптимизации выбора пути, а Datawise — на гибкости архитектуры. Будущее развития лежит в интеграции этих подходов для создания систем, которые могут одновременно быть высокоточными, надежными и адаптивными к изменяющимся условиям.''',
'Multi-Agent Frameworks and Collaboration': '''Это промежуточный аналитический артефакт, созданный в соответствии с требованиями научного синтеза. Он призван обеспечить основу для построения всеобъемлющей обзорной научной статьи по теме "multiagent systems of science automation".

## Обзор подтемы
Исследовательская область, охваемая этими работами, сосредоточена на разработке многоагентных систем (MAS) для автоматизации и ускорения научных открытий и процессов. Основная цель — преодоление ограничений традиционных систем, которые страдают от жесткости (фиксированные рабочие процессы), отсутствия стандартизации (интерфейсы к оборудованию) и неспособности к адаптации к динамическим научным контекстам.

Важность этого направления заключается в следующем: научные исследования требуют обработки огромных объемов разнородных данных, сложной многоэтапной координации (от планирования до эксперимента), и постоянной адаптации к новым результатам. Многоагентные системы предлагают архитектурный подход, позволяющий делегировать задачи специализированным агентам, обеспечивая гибкость, масштабируемость и потенциально более глубокое понимание контекста, что критически важно для создания "со-исследователей" (co-scientist systems).

## Основные идеи и методы
Представленные работы предлагают три основных, но взаимодополняющих подхода к решению проблемы автоматизации науки: динамическое управление процессами, стандартизация интерфейсов и обобщение знаний.

1. Динамические рабочие процессы и адаптивность (freephdlabor): Основная идея заключается в создании многоагентной структуры, способной к адаптации в реальном времени. Метод фокусируется на децентрализации контроля, где специализированные агенты управляют динамическими рабочими процессами, реагируя на текущие научные результаты. Это обеспечивает гибкость и возможность кастомизации систем.
2. Стандартизация и протоколы взаимодействия (SCP): Этот подход фокусируется на инфраструктурном уровне. Методология состоит в создании унифицированного протокольного слоя и стандартизированных интерфейсов для связи между различными научными ресурсами (модели, инструменты, оборудование). Это позволяет создать композируемую сеть автономных агентов, обеспечивая воспроизводимость и устраняя проблемы интеграции.
3. Обобщение и универсализация (ThinkTank): Этот метод направлен на повышение масштабируемости и переносимости систем. Идея заключается в создании фреймворка, который может обобщать знания, полученные в доменно-специфичных агентах, в универсальную платформу для совместного интеллекта. Это позволяет агентам эффективно взаимодействовать и адаптироваться в различных научных областях.

Сравнение подходов:
   freephdlabor фокусируется на логике процесса и внутренней коммуникации* агентов.
   SCP фокусируется на физической и инструментальной интеграции* агентов с внешним миром (оборудование, данные).
   ThinkTank фокусируется на когнитивной архитектуре и внешней совместимости* агентов (как они думают и взаимодействуют).

## Совпадающие выводы и подтверждающие результаты
Несмотря на различия в фокусе, все три работы сходятся в следующих ключевых выводах:

1. Необходимость гибкости и адаптивности: Все исследования признают, что традиционные, фиксированные рабочие процессы неэффективны для современного научного поиска. freephdlabor демонстрирует это через динамические рабочие процессы, а ThinkTank через способность агентов адаптироваться к новым задачам и доменам.
2. Важность многоагентного сотрудничества: Успех автоматизации зависит от способности агентов эффективно взаимодействовать. freephdlabor подчеркивает необходимость создания модульной архитектуры с рабочим пространством для коммуникации, а ThinkTank демонстрирует, что это сотрудничество может быть универсальным.
3. Устранение интеграционных проблем: SCP подтверждает, что для крупномасштабной автоматизации необходимо стандартизировать взаимодействие с внешними ресурсами (инструментами, данными). Это подтверждает, что архитектурная гибкость (freephdlabor) должна сочетаться с стандартизированным протоколом (SCP).

## Различия и противоречия
Основное расхождение между работами лежит в их уровне абстракции и целевом объекте автоматизации:

   Фокус на процессе vs. Фокус на интерфейсе: freephdlabor сосредоточен на динамике выполнения работы (как процесс меняется), в то время как SCP сосредоточен на интерфейсе (как агенты подключаются к оборудованию и данным). Разница заключается в том, что одна работа решает проблему управления процессом, а другая — проблему взаимодействия* с физической средой.
   Локальная адаптация vs. Глобальная обобщенность: freephdlabor стремится к адаптации конкретной системы к текущему исследовательскому контексту. ThinkTank стремится к обобщению знаний и адаптации агентов к разным доменам* (универсальная платформа). Это различие отражает переход от специализированной автоматизации к универсальному когнитивному интеллекту.
   Контекст vs. Структура: freephdlabor акцентирует внимание на управлении контекстом в процессе (memory/context management), в то время как SCP акцентирует внимание на управлении структурой* взаимодействия (protocol layer).

## Сильные стороны и ограничения
Каждый подход имеет свои уникальные преимущества и ограничения:

freephdlabor:
Сильные стороны: Обеспечивает высокую гибкость и кастомизацию, позволяет создавать специализированные системы "со-исследователей", включает человеко-в-цикле (human-in-the-loop), что критически важно для научного контроля.
Ограничения: Неявно подразумеваются сложности в управлении распределенными ограниченными ресурсами (capacity constraints) и масштабируемости для широкого доступа.

SCP:
Сильные стороны: Обеспечивает воспроизводимость и интеграцию, создавая единый протокольный слой для разнообразных ресурсов. Это решает проблему интеграции оборудования и данных, ускоряя воспроизводимость экспериментов.
Ограничения: Реализация может сталкиваться с проблемами совместимости с разнородным оборудованием (vendor-agnostic interfaces) и требует значительных усилий для стандартизации драйверов.

ThinkTank:
Сильные стороны: Предлагает архитектуру для универсального сотрудничества, позволяя агентам эффективно обобщать доменно-специфические знания и адаптироваться к новым задачам.
Ограничения: Потенциальные сложности, связанные с вычислительной сложностью, необходимостью большого объема обучающих данных и обеспечением робастности системы в сложных или непредсказуемых средах.

## Данные и экспериментальные оценки
В предоставленных описаниях акцент делается на архитектурной новизне и концептуальной реализации, а не на количественных экспериментальных метриках.

freephdlabor: Оценка сосредоточена на способности системы обеспечивать динамические рабочие процессы и модульную архитектуру. Оценка качества обеспечивается гибкостью и функциональностью системы для создания кастомизируемых систем.
SCP: Оценка сосредоточена на эффективности интеграции и воспроизводимости. Успех измеряется в способности системы обеспечивать бесшовное взаимодействие между вычислительными инструментами и физическими лабораторными инструментами.
ThinkTank: Оценка фокусируется на способности агентов к обобщению и обобщению знаний, что подразумевает оценку качества и широты их способности к переносу знаний между доменами.

## Заключение

Эти три направления (динамика, интеграция и обобщение) дополняют друг друга: динамика обеспечивает гибкость процесса, интеграция обеспечивает связь с реальным миром, а обобщение обеспечивает интеллектуальную основу. Будущие исследования должны стремиться к созданию систем, которые не только динамично управляют задачами, но и интегрируют эти задачи в реальные эксперименты, используя обобщенные знания.''',
'Scientific Discovery and Knowledge Integration': '''## Обзор подтемы
Данная подтема, "multiagent systems of science automation", охватывает пересечение области искусственного интеллекта (ИИ), многоагентных систем (MAS) и методологий научных исследований и разработки (SoS). Исследовательская область фокусируется на использовании автоматизации и агентных систем для улучшения междисциплинарного сотрудничества, интеграции знаний и автономного открытия научных закономерностей.

Важность направления обусловлена тем, что традиционные методы научного поиска часто ограничены человеческими когнитивными способностями и сложностью интеграции знаний между различными научными дисциплинами. Автоматизация, основанная на агентных системах, предлагает потенциал для преодоления этих ограничений, позволяя машинам не просто обрабатывать данные, но и участвовать в самостоятельном формировании и эволюции научных знаний. Это критически важно для ускорения темпов научных открытий и создания более комплексных, междисциплинарных научных теорий.

## Основные идеи и методы
Представленные работы предлагают два основных, но взаимосвязанных подхода к автоматизации науки: один фокусируется на интеграции знаний и человеческом сотрудничестве, а другой — на автономном открытии и анализе процессов.

1. Подход к интеграции знаний и сотрудничеству (CA-ChemE):
Основная идея: Использование многоагентных систем для облегчения межотраслевого сотрудничества и интеграции знаний между экспертами.
Ключевой метод: Инженерия онтологий (ontology engineering) в сочетании с архитектурой многоагентов (Collaboration Agent).
Цель: Преодоление семантических разрывов и различий в аналитических рамках между различными научными доменами.
Принцип работы: Агенты взаимодействуют, используя формальные структуры знаний (онтологии) для обеспечения согласованности и эффективной коммуникации.

2. Подход к автоматизации открытия (AI4SoS):
Основная идея: Использование ИИ для автоматизации процессов научного открытия и анализа эволюции научных сообществ (Science of Science).
Ключевой метод: Разработка пятиуровневой иерархической автономии для автоматизированного научного открытия.
Цель: Форсайт (прогнозирование) тенденций в технологиях и инновациях, анализ эволюции исследовательских сообществ и автоматизация процессов открытия научных паттернов.
Принцип работы: Агентные системы используются для анализа больших объемов данных и выявления скрытых причинно-следственных связей, что позволяет системе автономно формировать гипотезы и исследовать новые области.

Сравнение подходов:
В то время как подход CA-ChemE фокусируется на качестве и эффективности сотрудничества в рамках уже существующих экспертных отношений, подход AI4SoS фокусируется на количестве и автономии процесса открытия. Первый подход решает проблему коммуникации и интеграции знаний между людьми, а второй — проблему автоматизации самого процесса научного познания.

## Совпадающие выводы и подтверждающие результаты
Несмотря на различие в фокусе (сотрудничество vs. открытие), работы сходятся в следующем:

1.  Роль многоагентных систем: Обе работы демонстрируют, что архитектуры, основанные на агентах, являются эффективным механизмом для управления сложными научными задачами. В CA-ChemE агенты обеспечивают эффективное взаимодействие экспертов, а в AI4SoS — автономное исследование и прогнозирование.
2.  Необходимость интеграции знаний: Обе работы подчеркивают критическую необходимость преодоления семантических и методологических разрывов. CA-ChemE использует онтологии для интеграции знаний между доменами, а AI4SoS подразумевает интеграцию данных и паттернов для понимания эволюции науки.
3.  Повышение эффективности: Обе системы нацелены на повышение эффективности. В CA-ChemE это измеряется повышением эффективности сотрудничества экспертов, а в AI4SoS — повышением эффективности процесса научного открытия и анализа.
4.  Потенциал автоматизации: Обе работы видят в ИИ основу для автоматизации рутинных или сложных процессов в науке, что может привести к более быстрому и более глубокому научному прогрессу.

## Различия и противоречия
Основные различия между работами лежат в масштабе задачи, уровне автономии и цели взаимодействия.

1.  Фокус взаимодействия:
       CA-ChemE фокусируется на человеко-машинном* сотрудничестве и интеграции знаний между людьми (междисциплинарная коллаборация).
       AI4SoS фокусируется на машинно-автономном* открытии и анализе научных паттернов (автономное научное открытие).
2.  Механизм решения проблем:
       CA-ChemE использует структурирование* (онтологии) для устранения коммуникационных барьеров.
       AI4SoS использует прогнозирование и автономию* для выявления новых знаний и тенденций.
3.  Уровень автономии: CA-ChemE оперирует уровнем сотрудничества, где агенты помогают людям в совместном решении проблемы. AI4SoS оперирует уровнем автономного открытия, где система самостоятельно генерирует научные результаты и паттерны.

Объяснение различий: Различия объясняются тем, что первая работа рассматривает многоагентные системы как инструмент для улучшения человеческой когнитивной работы (интердисциплинарное взаимодействие), тогда как вторая рассматривает их как инструмент для машинной когнитивной работы (автономное научное познание).

## Сильные стороны и ограничения
CA-ChemE (Интеграция знаний):
Сильные стороны: Успешная демонстрация того, что онтологическая инженерия и многоагентные архитектуры могут эффективно преодолевать семантические разрывы между экспертами. Показывает конкретный путь к улучшению межотраслевого сотрудничества.
Ограничения: Система пока сталкивается с трудностями в открытом генерации гипотез и долгосрочном планировании исследований. Не достигает полной автономности.

AI4SoS (Автоматизация открытия):
Сильные стороны: Предлагает прогностическую и иерархическую структуру для автоматизации научного процесса. Позиционирует ИИ как основу для автоматизации процесса открытия научных закономерностей.
Ограничения: Признает младенческий этап автономного научного открытия. Существуют серьезные этические проблемы, связанные с ответственностью (accountability), предвзятостью (bias) и общественным доверием к решениям, принимаемым ИИ.

## Данные и экспериментальные оценки
   CA-ChemE: Оценка сосредоточена на эффективности сотрудничества*. Ключевой показатель — улучшение эффективности сотрудничества экспертных пар. Результаты показывают, что эффективность сотрудничества значительно улучшается, особенно при работе с доменами, находящимися на большом расстоянии (до 8.5% улучшения).
   AI4SoS: Оценка сосредоточена на потенциале автоматизации и прогнозировании*. Результаты демонстрируют потенциал ИИ в прогнозировании технологических тенденций и анализе эволюции исследовательских сообществ.

Сравнение оценки: CA-ChemE измеряет качество взаимодействия, тогда как AI4SoS измеряет потенциал автономного результата. Оба подхода используют агентные системы для достижения цели, но используют разные метрики успеха в зависимости от контекста задачи.

## Пробелы исследований и открытые вопросы
Несмотря на значительный прогресс, обе работы указывают на важные нерешенные вопросы:

1.  Масштаб автономности: Необходимо разработать системы, которые могут не просто помогать, но и самостоятельно генерировать сложные, открытые гипотезы и планировать долгосрочные исследовательские программы (проблема CA-ChemE).
2.  Каузальность и Причинно-следственные связи: Недостаточно изучено, как многоагентные системы могут эффективно моделировать и понимать причинно-следственные механизмы, лежащие в основе научного прогресса, а не только описывать корреляции.
3.  Этика и Доверие: Для систем, занимающихся автономным научным открытием, критически важно разработать механизмы для обеспечения справедливости (anti-bias), прозрачности и общественного доверия к результатам, полученным с помощью ИИ.
4.  Трансфер знаний: Требуется исследование потенциала этих агентных фреймворков для применения в других областях науки, помимо химической инженерии и Science of Science.

## Ключевые выводы для итогового мета-анализа
Для построения обзорной статьи по теме "multiagent systems of science automation" следует сохранить следующие ключевые выводы:

1.  Двойная роль MAS: Многоагентные системы играют двойную роль в науке: они могут быть эффективными инструментами для интеграции знаний (связывая людей и дисциплины) и автоматизации открытия (автономное формирование научных паттернов).
2.  Важность Структуры: Эффективность агентов напрямую зависит от использования формальных структур знаний (онтологий) для преодоления семантических барьеров и обеспечения согласованности.
3.  Разграничение Целей: Необходимо четко различать области применения MAS:
    *   Для коллаборации (CA-ChemE): Фокус на качестве взаимодействия и интеграции человеческих знаний.
    *   Для автономии (AI): Переход от помощи к автономному поиску и генерации новых знаний.
4.  Будущее: Развитие систем должно смещаться от простого анализа к генерации новых гипотез, требуя более сложных моделей, способных не только обрабатывать существующие данные, но и самостоятельно формулировать исследовательские вопросы.''',
'Scientific Workflow Automation': '''## Обзор подтемы
Данная исследовательская область фокусируется на автоматизации научных рабочих процессов (Scientific Workflow Automation) с использованием многоагентных систем (Multi-Agent Systems, MAS). Основная цель — преодоление сложности и трудоемкости выполнения многоэтапных научных задач, таких как моделирование, расчеты и анализ данных, путем делегирования этих задач интеллектуальным агентам.

Важность этого направления заключается в следующем:
1. Увеличение скорости открытия: Автоматизация сложных вычислительных задач резко сокращает время, необходимое для проведения экспериментов и анализа результатов (например, в материаловедении, физике и химии).
2. Масштабируемость: Многоагентные фреймворки позволяют управлять сложными, многоэтапными рабочими процессами, которые невозможно автоматизировать традиционными скриптами.
3. Использование ИИ: Это соединяет передовые возможности больших языковых моделей (LLM) и агентного мышления с высокопроизводительными вычислениями (HPC), создавая новый уровень автоматизации научных открытий.

## Основные идеи и методы
Исследуемые работы предлагают два основных подхода к автоматизации научных процессов:
1. Планирование и Оркестрация (TritonDFT): Этот подход фокусируется на высокоуровневом управлении сложными рабочими процессами. Агент выступает в роли планировщика, который использует экспертные знания для разбиения задачи на подзадачи, выбора оптимальных параметров (Pareto-aware parameter inference) и управления распределенными ресурсами (HPC orchestration).
2. Поэтапное Улучшение (Re4): Этот подход фокусируется на автоматизации самого процесса выполнения и улучшения научных результатов. Агент выполняет последовательные стадии работы с кодом, включая переписывание (rewriting), разрешение (resolution), обзор (review) и пересмотр (revision) научных вычислений.

Сравнение подходов:
TritonDFT — это система, сфокусированная на управлении потоком работы (workflow management) и ресурсами (resource management). Она использует агентов для принятия стратегических решений и оркестрации.
Re4 — это система, сфокусированная на улучшении качества вычислений (code quality and accuracy). Она использует агентов для итеративного улучшения самого научного кода и алгоритмов.

Группировка методов:
Оркестрация и Планирование: TritonDFT (использование LLM для планирования шагов).
Итеративное Улучшение: Re4 (использование многоступенчатых агентов для ревизии кода).
Интеграция LLM: Обе работы используют LLM в качестве основного исполнителя и планировщика, демонстрируя их способность к адаптивности к различным задачам.

## Совпадающие выводы и подтверждающие результаты
Несмотря на то что работы фокусируются на разных аспектах автоматизации, они подтверждают следующие общие выводы:
1. Эффективность автоматизации: Обе системы демонстрируют значительное ускорение и повышение эффективности научных задач по сравнению с ручным экспертным выполнением. TritonDFT показывает ускорение более чем в 10 раз.
2. Роль агентов: Многоагентные фреймворки являются эффективным инструментом для автоматизации сложных, многоэтапных научных задач, требующих сочетания различных типов экспертизы (планирование, расчеты, управление ресурсами).
3. Адаптивность LLM: Различные LLM демонстрируют разную эффективность в зависимости от конкретной задачи. В TritonDFT это проявляется в том, что GPT-5.2 превосходит в точности, Gemini 2.5 Flash — в соотношении точности и стоимости, а Opus 4.5 — в схемах параллелизации.

## Различия и противоречия
Основное различие между работами заключается в фокусе автоматизации:

| Аспект | TritonDFT | Re4 |
| :--- | :--- | :--- |
| Основная цель | Оркестрация всего научного рабочего процесса (от планирования до HPC). | Улучшение точности и эффективности самого научного кода и вычислений. |
| Уровень абстракции | Высокий (стратегическое планирование и управление ресурсами). | Низкий/Средний (детализированное изменение синтаксиса и логики кода). |
| Фокус внимания | Внешняя оптимизация (ресурсы, последовательность шагов). | Внутренняя оптимизация (код, алгоритмы, корректность). |
| Противоречие | TritonDFT сталкивается с проблемой точности при моделировании сложных квантовых состояний (магнитных материалов). | Re4 сталкивается с проблемой масштабируемости и вычислительной стоимости интеграции множественных этапов улучшения. |

Различия объясняются тем, что одна работа занимается инженерией процесса (как выполнить задачу максимально быстро и эффективно), а другая — инженерией содержания (как выполнить задачу максимально точно и корректно).

## Сильные стороны и ограничения
TritonDFT:
Сильные стороны: Отличное управление сложными, многомерными задачами; способность к оркестрации HPC; адаптивность к разным моделям (LLM); демонстрация огромного ускорения.
Ограничения: Ограниченная точность при работе со сложными физическими состояниями (например, магнитные материалы); открытые вопросы по расширению модульности для поддержки разнообразных решателей.

Re4:
Сильные стороны: Интегрированный подход к автоматизации всего цикла разработки кода (от ошибки до ревизии); комплексный метод улучшения качества науки.
Ограничения: Отсутствие явных количественных результатов; потенциальные проблемы масштабируемости при применении к крупномасштабным научным рабочим процессам; неопределенность в отношении вычислительной стоимости интегрированного подхода.

## Данные и экспериментальные оценки
Метрики: Основная метрика для TritonDFT — это ускорение (более 10x по сравнению с ручным выполнением). Для Re4 метрики, по-видимому, связаны с точностью и эффективностью научных вычислений.
Данныесеты/Сравнение: TritonDFT сравнивается с ручным экспертным выполнением и существующими инструментами, такими как инструменты для генерации скриптов (Mathew et al., 2017; Larsen et al., 2017) и инструменты управления HPC (Pizzi et al., 2016).
Сравнение моделей: В TritonDFT проводится сравнение производительности различных LLM (GPT-5.2, Gemini 2.5 Flash, Opus 4.5), что позволяет оценить, как архитектура агента взаимодействует с различными языковыми моделями.

## Пробелы исследований и открытые вопросы
На основе анализа выявлены следующие нерешенные вопросы:
1. Точность в сложных областях: Существует острая потребность в улучшении точности агентов при моделировании сложных физических состояний (например, магнитных материалов), что является критическим ограничением для TritonDFT.
2. Масштабируемость и Стоимость: Для Re4 ключевыми вопросами являются масштабируемость агента к очень крупным и сложным научным рабочим процессам, а также оценка вычислительной стоимости интегрированных многоступенчатых процессов.
3. Верификация и Доверие: Недостаточно изучены механизмы обеспечения надежной верификации и подтверждения результатов, сгенерированных многоагентными системами, что важно для научной достоверности.
4. Модульность: Необходимость расширения модульности фреймворков (как в TritonDFT) для поддержки широкого спектра разнообразных научных решателей и методов.

## Ключевые выводы для итогового мета-анализа
Для построения обзорной статьи следует выделить следующие ключевые тезисы:

1. Двухуровневая автоматизация: Автоматизация научных процессов должна рассматриваться на двух уровнях: оркестрация (управление потоком и ресурсами, как в TritonDFT) и улучшение содержания (автоматическое исправление и повышение качества кода и результатов, как в Re4).
2. Синергия Агентов и Экспертизы: Эффективность многоагентных систем достигается за счет интеграции различных типов экспертизы — стратегического планирования (TritonDFT) и детализированного кодового анализа (Re4).
3. Разделение Задач: Существует необходимость в разработке специализированных агентных фреймворков, оптимизированных либо для управления внешними ресурсами и потоками данных, либо для внутренней ревизии и коррекции научных данных.
4. Проблема Точности vs. Масштаб: Основные ограничения современных систем — это компромисс между достижением высокой точности в сложных доменах и обеспечением масштабируемости и вычислительной эффективности при автоматизации.
5. Будущее Направления: Будущие исследования должны сосредоточиться на разработке надежных механизмов верификации, повышении точности моделей в сложных областях и оценке вычислительной стоимости этих систем.'''
    }
    topic = 'multiagent systems of science automation'
    output_path = Path('analysis_output')
    output_dir='analysis_output'
    final_review = '''# Введение

## Введение

Современные научные исследования характеризуются экспоненциальным ростом объемов данных, сложностью многоэтапных процессов и необходимостью междисциплинарного сотрудничества. Однако традиционные методологии часто сталкиваются с ограничениями, связанными с хрупкостью рабочих процессов, недостаточной воспроизводимостью, сложностью интеграции разнородных знаний и ограниченностью человеческих когнитивных способностей к самостоятельному формированию сложных научных теорий. Для преодоления этих ограничений требуется архитектурный сдвиг, способный обеспечить не просто обработку данных, но и автономное выполнение сложных исследовательских задач.

В этом контексте многоагентные системы (Multi-Agent Systems, MAS) выступают как мощный методологический подход, предлагающий фреймворк для автоматизации и ускорения научных открытий. MAS позволяют делегировать сложные, многоэтапные задачи специализированным агентам, которые могут планировать, адаптироваться к динамическим контекстам и координировать действия на уровне, недоступном традиционным скриптовым системам.

Данный обзор фокусируется на области «multiagent systems of science automation», исследуя их применение в различных доменных областях науки и технологий. Мы синтезируем существующие исследования, охватывающие три ключевые оси: автоматизацию научных рабочих процессов, автоматизацию анализа данных и интеграцию знаний.

В частности, исследования в области автоматизации научных рабочих процессов (Scientific Workflow Automation) используют MAS для преодоления трудоемкости многоэтапных расчетов и моделирования, демонстрируя значительное увеличение скорости открытия и масштабируемость. В то же время, автоматизация науки о данных (Data Science Automation) применяет MAS, часто в связке с большими языковыми моделями (LLM), для решения проблем хрупкости, интерпретируемости и устойчивости при работе с гетерогенными и неструктурированными данными. Это подчеркивает потенциал MAS в обеспечении надежности и объяснимости результатов машинного обучения в научном контексте.

Кроме того, разработка многоагентных фреймворков и систем сотрудничества (Multi-Agent Frameworks and Collaboration) фокусируется на устранении жесткости традиционных систем и отсутствии стандартизации, предоставляя гибкую архитектуру для адаптации к меняющимся научным условиям. Это критически важно для создания систем, способных к динамическому планированию и взаимодействию, что открывает путь к созданию «со-исследовательских» систем.

Наконец, пересечение между агентными системами и методологиями научных исследований и разработки (Scientific Discovery and Knowledge Integration) исследует потенциал MAS для улучшения междисциплинарного сотрудничества и автономного открытия научных закономерностей. Здесь агентные системы рассматриваются не только как инструменты для выполнения задач, но и как механизмы для интеграции знаний между различными научными дисциплинами и ускорения процесса автономного формирования научных теорий.

Целью данного обзора является систематизация текущих достижений в области автоматизации науки с помощью многоагентных систем, выявление ключевых архитектурных подходов, анализ их применимости и определение перспектив для дальнейшего развития.

# Предметная область

## Предметная область: Многоагентные системы для автоматизации науки

Введение и обоснование области

Исследовательская область «многоагентные системы для автоматизации науки» (Multi-Agent Systems of Science Automation) находится на пересечении искусственного интеллекта, многоагентного мышления и методологий научных исследований и разработки (Science of Science, SoS). Основная цель этой области — разработка архитектур, способных не просто обрабатывать большие объемы данных, но и автономно управлять сложными, многоэтапными, междисциплинарными научными задачами, требующими планирования, адаптации, интеграции знаний и открытия новых закономерностей.

Традиционные методы научного поиска и анализа ограничены человеческими когнитивными способностями и сложностью интеграции знаний между различными научными дисциплинами. Автоматизация, основанная на агентных системах, предлагает архитектурный подход для преодоления этих ограничений, позволяя машинам участвовать в самостоятельном формировании и эволюции научных знаний.

Ключевые задачи и проблемы

Автоматизация науки сталкивается с рядом фундаментальных проблем, которые многоагентные системы призваны решить:

1. Управление сложностью и масштабируемость: Научные исследования часто включают многоэтапные рабочие процессы (от моделирования до эксперимента), которые невозможно эффективно автоматизировать традиционными скриптами. Агентные системы позволяют управлять сложными, многоуровневыми рабочими процессами, обеспечивая масштабируемость, необходимую для работы с огромными массивами данных и вычислений (HPC).
2. Воспроизводимость и надежность: В области науки о данных существует проблема хрупкости (brittleness) существующих методов, недостатка воспроизводимости и интерпретируемости результатов. Агенты могут обеспечить стабильность процесса, декомпозируя задачи и используя механизмы самокоррекции для обеспечения надежности вычислений.
3. Интеграция знаний и сотрудничество: Существующие научные методы страдают от семантических разрывов и трудностей интеграции знаний между различными научными доменами. Агентные системы предлагают архитектуру для облегчения междисциплинарного сотрудничества и интеграции знаний посредством формальных структур (например, онтологий), позволяя агентам взаимодействовать и согласовывать различные аналитические рамки.
4. Автономное открытие: Необходимость перехода от ручного анализа данных к автономному открытию научных закономерностей. Агентные системы могут выполнять сложные исследовательские задачи, интегрируя данные и формулируя гипотезы.

Основные направления исследований

Исследования в этой области сосредоточены на следующих аспектах:

1. Интеграция с данными и вычислительными задачами (Data-Intensive Tasks): Разработка систем, способных автономно управлять сложными вычислительными потоками, интегрируя данные из различных источников для решения научных проблем.
2. Управление рабочими процессами (Workflow Management): Создание фреймворков для управления многоэтапными исследовательскими задачами, где агенты самостоятельно планируют, выполняют и корректируют шаги.
3. Сотрудничество и Диалог (Collaboration and Reasoning): Разработка механизмов, позволяющих агентам взаимодействовать друг с другом и с экспертами для совместного анализа данных и генерации обоснованных выводов.
4. Интерпретация и Доверие (Interpretability and Trust): Обеспечение прозрачности решений, принимаемых агентами, для повышения доверия научных сообществ к результатам, полученным с помощью систем ИИ.

Таким образом, область исследований находится на стыке искусственного интеллекта, машинного обучения, информатики и методологии науки, стремясь к созданию интеллектуальных систем, способных к автономному и высокоэффективному научному познанию.

# Методы и Подходы

## Методы и Подходы

Автоматизация науки посредством многоагентных систем (MAS) требует интеграции различных методологий, охватывающих как внутреннюю архитектуру агентов, так и внешнее взаимодействие с научными рабочими процессами и знаниями. Представленные исследования можно сгруппировать по трем основным осям: планирование и оркестрация рабочих процессов, архитектура и адаптивность агентов, а также интеграция знаний и сотрудничество.

1. Планирование, Оркестрация и Итеративное Улучшение Рабочих Процессов

Автоматизация научных процессов часто фокусируется на управлении сложным потоком работы (workflow management) и повышении качества вычислений.

Оркестрация и Планирование: Подход, сфокусированный на высокоуровневом управлении сложными задачами, использует агентов в качестве планировщиков. Методы, такие как использование LLM для многопутевого планирования (Multi-path planning) и ансамблевой оптимизации (SPIO), позволяют агентам исследовать пространство возможных стратегий, выбирая наиболее надежные пути для достижения цели. В то же время, системы оркестрации (например, TritonDFT) используют агентов для принятия стратегических решений и управления распределенными ресурсами (HPC orchestration), что позволяет эффективно управлять многомерными задачами и распределенными вычислительными ресурсами.

Поэтапное Улучшение: Другой подход фокусируется на автоматизации самого процесса выполнения и итеративном улучшении научных результатов. Системы, подобные Re4, используют многоступенчатые агенты для автоматизации всего цикла разработки кода, включая переписывание, разрешение и пересмотр научных вычислений. Этот метод акцентирует внимание на автоматизации улучшения качества кода и алгоритмов, интегрируя процесс научного открытия в итеративный цикл.

2. Архитектура, Стабильность и Адаптивность Агентов

Ключевым аспектом автоматизации является разработка архитектур, обеспечивающих стабильность, робастность и способность агентов к адаптации в динамических научных средах.

Стабильность и Воспроизводимость: Для обеспечения воспроизводимости научных процессов разрабатываются подходы, основанные на структурированном рассуждении. Методы, такие как PCS-Guided (Principles of Science-Guided) подход в VDSAgents, декомпозируют конвейер науки о данных на последовательные шаги, что снижает риск ошибок.

Интеграция и Ресурсы: Для обеспечения взаимодействия с внешними системами, существуют подходы, которые фокусируются на создании общих фреймворков. Это включает разработку архитектур, которые могут интегрировать различные знания и ресурсы, что критически важно для систем, работающих в сложных научных доменах.

3. Интеграция и Экосистемы

Эффективная реализация сложных систем требует создания надежных сред для взаимодействия между компонентами. Это включает разработку протоколов и архитектур, которые позволяют различным агентам и системам обмениваться информацией и выполнять задачи в единой экосистеме.

Резюме

Данный обзор демонстрирует, что успешная автоматизация научных процессов требует комплексного подхода, сочетающего методы оптимизации (планирование и итерация), создание надежных архитектур (интеграция и управление ресурсами) и фокусировку на знаниях (интеграция и коллаборация).

# Ключевые Результаты и Открытия

## Ключевые Результаты и Открытия

Многоагентные системы (MAS) представляют собой мощный фреймворк для автоматизации сложных научных процессов, демонстрируя потенциал для ускорения открытия знаний и повышения эффективности научного рабочего процесса. Анализ результатов различных исследований выявляет несколько ключевых направлений и противоречий, определяющих будущее этой области.

1. Улучшение производительности и робастность систем

Исследования подтверждают, что применение многоагентных систем позволяет превосходить производительность традиционных базовых систем. Системы демонстрируют способность не только выполнять задачи, но и обеспечивать высокую стабильность и устойчивость в условиях неопределенности.

Оптимизация и стратегия: В контексте анализа данных и прогнозирования, агенты успешно реализуют многоэтапное планирование и ансамблирование стратегий. Это указывает на то, что успех автоматизации зависит от способности системы проводить последовательное исследование и адаптивное принятие решений, а не просто следовать фиксированным алгоритмам.
Робастность и модульность: Для достижения высокой надежности критически важны механизмы восстановления ошибок и модульная архитектура. Разработки подчеркивают необходимость внедрения модулей планирования и исправления кода, что позволяет агентам адаптироваться к ошибкам и обеспечивать воспроизводимость результатов, особенно при работе со сложными и зашумленными наборами данных.

2. Архитектурные требования к многоагентному сотрудничеству

Успешная автоматизация сложных научных задач требует перехода от фиксированных рабочих процессов к гибким, адаптивным архитектурам, основанным на эффективном взаимодействии агентов.

Гибкость и адаптивность: Традиционные, статичные рабочие процессы признаны неэффективными для современного научного поиска. Агенты демонстрируют способность к динамической адаптации к новым задачам и доменам, что требует создания модульной архитектуры с рабочим пространством для коммуникации.
Интеграция взаимодействия: Для крупномасштабной автоматизации необходимо устранение интеграционных проблем. Это означает, что архитектурная гибкость должна сочетаться со стандартизированными протоколами взаимодействия с внешними ресурсами (данными, инструментами), что обеспечивает эффективное сотрудничество между агентами и физической средой.

3. Разграничение целей автоматизации: Процесс против Содержания

Существуют заметные различия в фокусе многоагентных систем, которые можно разделить на две основные категории: инженерия процесса (Workflow Automation) и инженерия содержания (Scientific Discovery).

Инженерия процесса (Workflow Automation): Некоторые системы фокусируются на внешней оптимизации — максимальном ускорении и эффективности выполнения задачи (например, оркестрация ресурсов и последовательность шагов). Здесь акцент делается на управлении внешними ресурсами и оптимизации вычислительного процесса.
Инженерия содержания (Scientific Discovery): Другие системы направлены на внутреннюю оптимизацию и открытие новых знаний. Здесь агенты используются для автономного исследования, анализа паттернов и прогнозирования, что требует более высокого уровня когнитивной автономии.

4. Роль интеграции знаний и автономии

Многоагентные системы выступают как эффективный механизм для преодоления семантических и методологических разрывов, что является критически важным для интеграции знаний в науке.

Интеграция знаний: Для успешного научного прогресса необходимо не только автоматизировать расчеты, но и интегрировать знания из различных доменов. Это достигается через использование онтологий и структур для управления коммуникацией и интеграции информации между экспертами и машинами.
Сотрудничество против автономии: Исследования показывают, что архитектуры агентов могут служить как для улучшения человеческой когнитивной работы (через междисциплинарное сотрудничество экспертов), так и для реализации машинного когнитивного открытия (через автономное прогнозирование и анализ). Это подчеркивает, что многоагентные системы могут быть как инструментом для поддержки человеческого познания, так и самостоятельным механизмом научного познания.

5. Сравнительные выводы о моделях автоматизации

Различия в результатах отражают фундаментальное расхождение в том, что именно автоматизируется:

| Аспект | Фокус на процессе (Workflow) | Фокус на результате (Discovery) |
| :--- | :--- | :--- |
| Цель | Эффективное выполнение заданных операций. | Открытие новых знаний и теорий. |
| Роль Агентов | Исполнители и координаторы. | Исследователи и генераторы гипотез. |
| Ключевой Вызов | Оптимизация рабочего процесса. | Интерпретация и валидация результатов. |

Таким образом, при разработке систем искусственного интеллекта для науки необходимо различать задачи: оптимизацию процедур (Data-to-Insight) и генерацию новых знаний (Insight-to-Discovery).

# Ограничения и Проблемы

## Ограничения и Проблемы

Развитие многоагентных систем для автоматизации научных процессов сталкивается с рядом фундаментальных ограничений, которые затрагивают технические, архитектурные и эпистемологические аспекты. Эти проблемы возникают на пересечении задач оптимизации рабочих процессов, обеспечения стабильности данных, управления распределенными системами и достижения машинного познания.

1. Проблемы стабильности, оптимальности и точности

Существует противоречие между стремлением к стабильности процесса и поиском глобально оптимальных решений. Методологии, ориентированные на стабильность и воспроизводимость (например, VDSAgents), фокусируются на методологически обоснованном процессе (PCS) и оценке чувствительности данных. В то же время, подходы, ориентированные на стратегию (SPIO), используют многопутевое планирование и ансамблевое обучение для поиска оптимальных стратегий, признавая, что многопутевое планирование не гарантирует глобально оптимальных конвейеров. Это создает дилемму: как совместить надежное и интерпретируемое выполнение задачи с эффективным поиском наилучшего результата.

Кроме того, в контексте автоматизации научных рабочих процессов наблюдается различие между инженерией процесса и инженерией содержания. Автоматизация рабочего процесса (TritonDFT) фокусируется на внешней оптимизации — оркестрации ресурсов и последовательности шагов, сталкиваясь с ограничениями точности при моделировании сложных физических состояний. В то же время, автоматизация содержания (Re4) фокусируется на внутренней оптимизации — улучшении точности самого научного кода и алгоритмов. Это указывает на необходимость разработки систем, способных одновременно управлять вычислительными ресурсами и обеспечивать научную корректность результатов.

2. Архитектурные и интеграционные вызовы

Создание эффективных многоагентных систем требует сложной архитектуры для управления непредсказуемыми данными и многоступенчатыми процессами. Это приводит к проблемам масштабируемости и управлению сложностью. В то время как одни системы стремятся к комплексной архитектуре для обработки непредсказуемых данных через модули планирования и выполнения (Datawise), другие сталкиваются с необходимостью управления распределенными ограниченными ресурсами и масштабируемостью для широкого доступа (freephdlabor).

Интеграция разнообразных ресурсов и систем также представляет серьезную проблему. Хотя фреймворки, такие как SCP, предлагают единый протокольный слой для интеграции оборудования и данных, они сталкиваются с проблемами совместимости с разнородным оборудованием и требуют значительных усилий для стандартизации интерфейсов. Это подчеркивает необходимость разработки универсальных и агностических интерфейсов, способных эффективно управлять взаимодействием между агентами и физической средой.

3. Проблемы автономии и эпистемологической ответственности

Ключевые ограничения многоагентных систем связаны с уровнем автономии и механизмом познания. Существует различие между системами, ориентированными на сотрудничество и интеграцию знаний, и системами, ориентированными на автономное открытие.

В контексте сотрудничества (CA-ChemE) агенты используются для преодоления семантических разрывов между экспертами, фокусируясь на улучшении междисциплинарного взаимодействия. Однако для достижения полной автономности научного открытия (AI4SoS) возникают более глубокие проблемы. Автономное открытие требует прогностической и иерархической структуры, но при этом порождает серьезные этические проблемы, связанные с ответственностью (accountability), предвзятостью (bias) и общественным доверием к решениям, принимаемым ИИ. Таким образом, основная проблема заключается в переходе от автоматизации человеко-машинного сотрудничества к полностью автономному научному познанию, требующему новых механизмов для обеспечения прозрачности и ответственности.

4. Различия в фокусе автоматизации

Различия в целях автоматизации также определяют ограничения систем. Некоторые подходы сосредоточены на управлении динамикой выполнения работы и контекстом (freephdlabor), тогда как другие стремятся к глобальной обобщенности знаний и адаптации агентов к различным доменам (ThinkTank). Это различие отражает переход от специализированной автоматизации к универсальному когнитивному интеллекту. В конечном счете, ограничения многоагентных систем определяются необходимостью балансировать между технической эффективностью (скорость и точность), архитектурной интеграцией (масштабируемость) и эпистемологическими целями (качество и интерпретируемость результатов).

# Будущие Направления

## Будущие Направления

Развитие многоагентных систем для автоматизации науки (Multiagent Systems of Science Automation) находится на критическом этапе перехода от выполнения рутинных вычислительных задач к автономному научному открытию и интеграции знаний. Анализ текущих пробелов в исследованиях выявляет необходимость одновременного решения технических, когнитивных и этических проблем, которые определяют потенциал этих систем.

1. Углубление технической точности и масштабируемости

Ключевым направлением для повышения применимости многоагентных систем является преодоление фундаментальных технических ограничений, выявленных в области автоматизации научных рабочих процессов. Существует острая потребность в разработке агентов, способных работать с высокой точностью в сложных, многомерных областях, таких как моделирование физических состояний (например, магнитных материалов). Это требует создания архитектур, которые могут интегрировать высокоточные научные решатели с агентными механизмами, обеспечивая при этом масштабируемость. Необходимо разработать новые методы для оценки и управления вычислительной стоимостью интегрированных многоступенчатых процессов, чтобы обеспечить эффективное масштабирование агентов к очень крупным и сложным исследовательским задачам. Кроме того, для обеспечения гибкости научных фреймворков требуется значительное расширение модульности, позволяющее агентам взаимодействовать с широким спектром разнообразных научных решателей и методологий.

2. Переход к автономному научному открытию и причинно-следственному моделированию

Второй стратегический вектор развития фокусируется на расширении когнитивных возможностей многоагентных систем, переходя от роли исполнителей к роли автономных исследователей. Необходимо разработать архитектуры, способные не просто выполнять заданные рабочие процессы, но и самостоятельно генерировать сложные, открытые научные гипотезы и планировать долгосрочные исследовательские программы, что является критическим шагом для решения задач автономного научного открытия.

Параллельно с этим, существует острая необходимость в развитии механизмов для моделирования и понимания причинно-следственных связей в научном прогрессе. Текущие системы, преимущественно основанные на корреляционном анализе, требуют доработки для эффективного моделирования глубинных причинно-следственных механизмов, лежащих в основе научного прогресса, а не только описания статистических корреляций. Развитие этих возможностей позволит многоагентным системам перейти от описания данных к формированию глубокого научного понимания.

3. Обеспечение верификации, прозрачности и этики

Поскольку многоагентные системы будут играть всё более значимую роль в принятии научных решений, критически важным становится обеспечение механизмов надежной верификации и доверия к результатам, генерируемым этими системами. Это включает разработку формальных механизмов для подтверждения научного обоснования и достоверности результатов, а также механизмов обеспечения прозрачности (explainability) в процессе принятия решений агентами.

Одновременно с этим, для систем, занимающихся автономным научным открытием и интеграцией знаний, требуется разработка комплексных этических рамок. Необходимо создать механизмы для обеспечения справедливости (anti-bias), прозрачности и общественного доверия к результатам, полученным с помощью ИИ. Это подразумевает исследование того, как структурировать взаимодействие агентов таким образом, чтобы минимизировать смещения и обеспечить соответствие научным и социальным стандартам.

4. Трансфер знаний и общее применение

Наконец, для максимизации общего научного воздействия многоагентные фреймворки должны быть расширены за рамки узкоспециализированных областей (например, химической инженерии и Science of Science). Необходимо исследовать потенциал этих агентных архитектур для трансфера знаний и методологий в другие научные дисциплины, что позволит создать универсальные инструменты для автоматизации и интеграции знаний в широком научном сообществе. Это потребует разработки абстрактных представлений знаний, которые могут быть применены вне зависимости от специфики исходного научного домена.

# Заключение

## Заключение

Многоагентные системы (МАС) представляют собой мощный методологический подход к автоматизации научных процессов, играя двойную роль: они выступают как инструменты для интеграции знаний и коллаборации, так и как движущая сила для автономного открытия и генерации новых научных паттернов. Анализ текущих исследований позволяет выделить несколько ключевых аспектов и определить вектор развития данной области.

Во-первых, эффективность МАС зависит от правильного архитектурного разделения задач. Для достижения высокой производительности необходимо реализовать двойную автоматизацию: оркестрацию внешних ресурсов и потоков данных (управление рабочим процессом) и улучшение содержания (автоматическое исправление, верификация и повышение качества научных результатов). Это требует разработки специализированных агентных фреймворков, оптимизированных либо для управления внешними потоками, либо для внутренней ревизии и коррекции данных.

Во-вторых, критически важным условием для функционирования сложных систем является интеграция различных типов экспертизы. Эффективность многоагентных систем достигается за счет синергии между стратегическим планированием (управление потоками и ресурсами) и детализированным анализом (коррекция и повышение точности данных). Для преодоления семантических барьеров и обеспечения согласованности знаний, как подчеркивается в контексте интеграции знаний, необходимо использовать формальные структуры, такие как онтологии, в качестве основы для агентных взаимодействий.

В-третьих, существует фундаментальный компромисс между точностью и масштабируемостью. Современные системы сталкиваются с ограничением, связанным с необходимостью достижения высокой точности в сложных научных доменах при одновременном обеспечении вычислительной эффективности и масштабируемости автоматизации. Это ставит задачу разработки более надежных механизмов верификации и оценки вычислительной стоимости многоагентных решений.

В заключение, будущее многоагентных систем в науке лежит в смещении фокуса от простого анализа существующих данных к автономной генерации новых гипотез и исследовательских вопросов. Будущие исследования должны сосредоточиться на создании более сложных моделей, способных не только обрабатывать существующие знания, но и самостоятельно формулировать исследовательские цели. Это потребует разработки систем, способных к самокоррекции, повышению точности моделей и обеспечению прозрачности процесса открытия, тем самым трансформируя МАС из инструментов автоматизации в полноценные агенты научного открытия.

# Гипотезы и дальнейшие исследования

Пожалуйста, предоставьте раздел "Гипотезы и дальнейшие исследования", который вы хотите, чтобы я проверил. Я применю роль строгого научного критика, чтобы оценить новизну, актуальность и потенциальный вклад ваших идей в область исследования.'''
    papers_info = load_extracted_info([ArxivPaper('2510.24339v2', None, None, None, None, None, None, None, None), ArxivPaper('2503.23314v2', None, None, None, None, None, None, None, None), ArxivPaper('2503.07044v2', None, None, None, None, None, None, None, None), ArxivPaper('2603.03372v2', None, None, None, None, None, None, None, None), ArxivPaper('2508.20729v2', None, None, None, None, None, None, None, None), ArxivPaper('2510.15624v1', None, None, None, None, None, None, None, None), ArxivPaper('2512.24189v1', None, None, None, None, None, None, None, None), ArxivPaper('2506.02931v1', None, None, None, None, None, None, None, None), ArxivPaper('2510.01293v1', None, None, None, None, None, None, None, None), ArxivPaper('2505.12039v1', None, None, None, None, None, None, None, None)])

    print("[DEBUG] Генерирую раздел гипотез и дальнейших исследований")
    hypotheses_text = generate_hypotheses_section(group_analyses, papers_info, topic, num_iterations=2)
    print(f"[DEBUG] Текст раздела гипотез: {hypotheses_text}\n")
    final_review = final_review.strip() + "\n\n# Гипотезы и дальнейшие исследования\n\n" + hypotheses_text

    input()

    # Сохраняем обзор без ссылок, уже с разделом гипотез
    with open(output_path / 'final_review_no_citations.txt', 'w', encoding='utf-8') as f:
        f.write(final_review)

    # Добавляем ссылки на статьи
    print("[DEBUG] Добавляю ссылки на статьи в обзор")
    final_review_with_citations = add_citations_to_review(final_review, papers_info, topic, num_iterations=3)
    
    # Сохраняем итоговый обзор со ссылками
    with open(output_path / 'final_review.txt', 'w', encoding='utf-8') as f:
        f.write(final_review_with_citations)
    
    print(f"[DEBUG] Анализ завершен. Результаты сохранены в {output_dir}")