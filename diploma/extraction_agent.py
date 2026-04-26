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

Исправь группировку: все статьи должны быть распределены ровно по одной группе без дубликатов. Суммарно групп должно быть не больше 4.

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
    Генерирует анализ для одной группы статей с итеративным улучшением (компактные промпты).
    """
    group_papers = [p for p in papers_info if p['arxiv_id'] in paper_ids]
    
    papers_text = []
    for paper in group_papers:
        text = f"Title: {paper['title']}\nProblem: {paper['problem']}\nCompared baselines: {paper['compared_baselines']}\nResults: {paper['results']}\nLimitations: {paper['limitations']}\nNovelty: {paper['novelty']}\nkey_findings: {paper['key_findings']}\nopen_questions: {paper['open_questions']}"
        papers_text.append(text)
    papers_combined = "\n\n".join(papers_text)
    
    analysis = None
    
    for iteration in range(num_iterations):
        if iteration == 0:
            # Первичная генерация анализа
            msg_history = []
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
            # Проверка качества
            if analysis is None:
                continue
            
            check_prompt = f"""
Проверь анализ подтемы "{group_name}" на полноту, точность, согласованность, relevance к статьям, научную строгость и ясность структуры. Дай подробную обратную связь по любым слабым местам, несоответствиям или областям для улучшения.
Проверь:
1. Все ли статьи действительно отражены.
2. Достаточно ли глубок анализ.
3. Есть ли сравнение, противоречия и связи.
4. Все ли утверждения имеют ссылки на статьи.
5. Выявлены ли пробелы исследований.
6. Достаточно ли качественный это промежуточный артефакт для итогового обзора.

Анализ:
{analysis}

Резюме статей:
{papers_combined}

Если анализ хороший, ответь только "ПРИНЯТО". В противном случае, предоставь конкретные, действенные предложения для улучшения.
"""
            
            check_response, msg_history = get_response_from_llm(
                msg=check_prompt,
                print_debug=False,
                msg_history=msg_history,
                rate_limit=OPENROUTER_API_EXTRACT_RATE_LIMIT_SEC
            )
            
            if "ПРИНЯТО" in check_response.upper():
                break
            else:
                # Компактное улучшение
                improve_prompt = f"""
Вот текущий анализ подтемы "{group_name}":
{analysis}

Вот фидбек для улучшения:
{check_response}

Сделай новую исправленную версию анализа, учитывая фидбек. Сохрани сильные части, усиль слабые разделы, добавь недостающий сравнительный анализ, исправь ссылки на статьи, повышай ценность для итогового обзора.
"""
                
                response, _ = get_response_from_llm(
                    msg=improve_prompt,
                    print_debug=False,
                    rate_limit=OPENROUTER_API_EXTRACT_RATE_LIMIT_SEC
                )
                analysis = response
    
    return analysis

def generate_final_review(group_analyses, topic, num_iterations=3):
    """
    Генерирует итоговый обзор статьи по фиксированным разделам с targeted контекстом.
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
    
    def extract_relevant_parts(analyses, relevant_headers):
        """Извлекает релевантные части из анализов по заголовкам."""
        extracted = []
        for group_name, analysis in analyses.items():
            lines = analysis.split('\n')
            current_section = ""
            for line in lines:
                if any(header in line for header in relevant_headers):
                    current_section = line.strip()
                elif current_section and line.strip() and not line.startswith('#'):
                    extracted.append(f"Из группы '{group_name}': {line.strip()}")
        return '\n'.join(extracted) 
    
    # Генерировать каждый раздел
    final_sections = {}
    for section_name in sections:
        print(f"[DEBUG] Генерирую раздел: {section_name}")

        relevant_parts = extract_relevant_parts(group_analyses, section_mapping[section_name])
        print(f"[DEBUG] Извлеченные релевантные части для раздела {section_name}: \n {relevant_parts}\n")
        
        section_content = None
        
        for iteration in range(num_iterations):
            if iteration == 0:
                # Первичная генерация раздела
                prompt = f"""
Напиши раздел "{section_name}" научной обзорной статьи по теме "{topic}".

Используй только эту релевантную информацию из анализов групп:
{relevant_parts}

Требования:
- Научный стиль, синтез (не пересказ).
- Сравнивай подходы между группами.
- Сохраняй ссылки на статьи [CITATION: arxiv_id | название].
- Учитывай противоречия и консенсус.
- Раздел уровня обзорной публикации.
"""
                response, _ = get_response_from_llm(
                    msg=prompt,
                    print_debug=False,
                    rate_limit=OPENROUTER_API_EXTRACT_RATE_LIMIT_SEC
                )
                section_content = response
            else:
                # Компактное улучшение
                check_prompt = f"""
Проверь раздел "{section_name}" обзорной статьи по теме "{topic}".
Оцени полноту, согласованность, логичность, научную строгость, глубину, ссылки на статьи, не потеряна ли информация из анализов групп.

Раздел:
{section_content}

Релевантная информация:
{relevant_parts}

Дай детальную обратную связь. Если удовлетворительно, скажи "ПРИНЯТО". Иначе - конкретные предложения.
"""
                
                check_response, _ = get_response_from_llm(
                    msg=check_prompt,
                    print_debug=False,
                    rate_limit=OPENROUTER_API_EXTRACT_RATE_LIMIT_SEC
                )
                
                if "ПРИНЯТО" in check_response.upper():
                    break
                else:
                    improve_prompt = f"""
Вот текущий раздел "{section_name}":
{section_content}

Вот фидбек:
{check_response}

Сделай новую исправленную версию раздела, учитывая фидбек.
"""
                    
                    response, _ = get_response_from_llm(
                        msg=improve_prompt,
                        print_debug=False,
                        rate_limit=OPENROUTER_API_EXTRACT_RATE_LIMIT_SEC
                    )
                    section_content = response
        
        final_sections[section_name] = section_content
    
    # Собрать итоговую статью
    full_review = "\n\n".join([f"# {section_name}\n\n{content}" for section_name, content in final_sections.items()])

    print(f'[DEBUG] Полный обзор до финальной проверки: \n {full_review} \n')
    
    # Финальная проверка всей статьи
    check_prompt = f"""
Проверь целиком обзорную статью по теме "{topic}".

Проверь:
- согласованность разделов
- повторы
- пропущенные переходы
- противоречия
- недостающий синтез
- полноту раздела с пробелами исследований

Статья:
{full_review}

Улучши статью глобально, не ломая качественные части. Если удовлетворительно, скажи "ПРИНЯТО". Иначе - исправленная версия.
"""
    
    final_check, _ = get_response_from_llm(
        msg=check_prompt,
        print_debug=False,
        rate_limit=OPENROUTER_API_EXTRACT_RATE_LIMIT_SEC
    )
    
    if "ПРИНЯТО" in final_check.upper():
        return full_review
    else:
        return final_check  


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

Сформируй подробный анализ со следующими разделами:
1. Обзор подтемы: Какую исследовательскую область покрывают работы? Почему направление важно?
2. Основные идеи и методы: Какие подходы предлагаются? Сгруппируй методы по типам. Сравнивай подходы между собой.
3. Совпадающие выводы и подтверждающие результаты: Какие идеи подтверждаются несколькими работами? Указывай какие статьи это подтверждают.
4. Различия и противоречия: Где статьи расходятся? Чем объясняются различия?
5. Сильные стороны и ограничения: Сравни преимущества и недостатки подходов. Отрази ограничения, упомянутые в работах.
6. Данные и экспериментальные оценки: Какие датасеты, метрики и способы оценки используются? Что общего и что различается?
7. Пробелы исследований и открытые вопросы: Что остается нерешенным? Какие направления изучены недостаточно?
8. Ключевые выводы для итогового мета-анализа: Какие выводы важно сохранить для дальнейшего построения обзорной статьи?

ТРЕБОВАНИЯ:
- Анализ должен быть подробным и вдумчивым.
- Это промежуточный материал для финальной обзорной статьи.
- Каждое содержательное утверждение сопровождай ссылкой вида:
[CITATION: arxiv_id | название статьи]
- Не делай утверждений без опоры на статьи.
- Делай акцент на сравнении, синтезе и связях.

Статьи в этой группе:
{papers_combined}

Пиши анализ в структурированном формате с четкими заголовками.
"""


# Промпт для проверки качества анализа группы статей
check_group_analysis_prompt = """
Проверь анализ подтемы "{group_name}" на полноту, точность, согласованность, relevance к статьям, научную строгость и ясность структуры. Дай подробную обратную связь по любым слабым местам, несоответствиям или областям для улучшения.
Проверь:
1. Все ли статьи действительно отражены.
2. Достаточно ли глубок анализ.
3. Есть ли сравнение, противоречия и связи.
4. Все ли утверждения имеют ссылки на статьи.
5. Выявлены ли пробелы исследований.
6. Достаточно ли качественный это промежуточный артефакт для итогового обзора.

Если анализ хороший, ответь только "ПРИНЯТО". В противном случае, предоставь конкретные, действенные предложения для улучшения: что пропущено, что поверхностно, какие утверждения не подтверждены, как улучшить анализ.
Отвечай либо "ПРИНЯТО", либо подробными предложениями для улучшения.
"""


# Промпт для улучшения анализа группы статей
improve_group_analysis_prompt = """
Основываясь на обратной связи написанного тобой анализа, улучши анализ подтемы "{group_name}".

Фидбек на текущий анализ:
{check_response}

Предоставь улучшенную версию анализа, учитывая обратную связь: сохрани сильные части, усиль слабые разделы. Расширь недостаточно подробные части. Добавь недостающий сравнительный анализ. Исправь или добавь ссылки на статьи. Повышай ценность анализа как промежуточного артефакта.
"""


# Глобальная проверка всей обзорной статьи
final_review_check_prompt = """ 
Отредактируй следующую обзорную статью как цельный научный обзор. Верни только итоговый текст.

Проверь и при необходимости улучши статью по следующим критериям:
- согласованность разделов
- повторы
- пропущенные переходы
- противоречия
- недостающий синтез
- полноту раздела с пробелами исследований

Текст статьи:
{review}

Улучши статью глобально, не ломая уже качественные части.
Верни ТОЛЬКО финальную обзорную статью. Не включай комментарии, не включай оценку качества, не включай пояснения. Только текст статьи.
"""


if __name__ == "__main__":
    pass
    # txt_dir = "txts"
    # output_dir = "extracted_info"
    # extract_key_info_from_papers([ArxivPaper(arxiv_id="2409.11363v1", title="Test Paper", abstract="Test abstract", authors=[],published=None, updated=None, doi=None, pdf_url=None, source_url=None)], output_dir=output_dir, txt_dir=txt_dir)
