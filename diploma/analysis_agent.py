import json
from pathlib import Path

from llm import get_response_from_llm
from extraction_agent import extract_json_from_response
from utils import load_logger, extract_response_block, load_extracted_info

OPENROUTER_API_EXTRACT_RATE_LIMIT_SEC = 5.0 


logger = load_logger()


def group_papers_by_subtopics(topic, extracted_info_from_papers):
    '''
    Группирует статьи по подтемам с помощью LLM и итеративным улучшением.
    Возвращает словарь: {group_name: [paper_ids]}
    '''
   
    papers_summary = []
    for paper_info in extracted_info_from_papers:
        summary = f"Title: {paper_info['title']}\nProblem: {paper_info['problem']}\nCompared baselines: {paper_info['compared_baselines']}\nResults: {paper_info['results']}"
        papers_summary.append(f"Paper {paper_info['arxiv_id']}:\n{summary}")
    
    papers_summary_text = "\n\n".join(papers_summary)
    logger.info(f"Сформирован текст для группировки статей по подтемам:\n{papers_summary_text}\n")  
    
    # Первичная группировка
    groups = None
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
    logger.info(f"Результат первичной группировки: {groups}\n")

    # Проверка и улучшение
    all_paper_ids = set()
    duplicates = []
    for group_papers in groups.values():
        for paper_id in group_papers:
            if paper_id in all_paper_ids:
                duplicates.append(paper_id)
            all_paper_ids.add(paper_id)
    
    expected_papers = {p['arxiv_id'] for p in extracted_info_from_papers}
    missing = expected_papers - all_paper_ids
    extra = all_paper_ids - expected_papers
    
    if missing or duplicates or extra:
        # Фидбек для улучшения
        feedback = ""
        if missing:
            feedback += f"Не распределены статьи: {list(missing)}. "
        if duplicates:
            feedback += f"Дубликаты статей: {duplicates}. "
        if extra:
            feedback += f"Лишние статьи: {list(extra)}. "
        
        logger.info(f'Фидбек для улучшения групп: {feedback}')

        response, _ = get_response_from_llm(
            msg=improve_grouping_prompt.format(
                groups=json.dumps(groups, indent=2),
                feedback=feedback,
            ),
            print_debug=False,
            temperature=0.1,
            rate_limit=OPENROUTER_API_EXTRACT_RATE_LIMIT_SEC
        )
        groups = extract_json_from_response(response)
        logger.info(f"Результат повторной группировки: {groups}\n")
    
    if not groups:
        logger.error("Не удалось распарсить группы из ответа LLM")
        return {}
    
    return groups


def generate_group_analysis(extracted_info_from_papers, group_name, paper_ids, topic, num_iterations=3):
    '''
    Генерирует анализ для одной группы статей с итеративным улучшением.
    '''
    group_papers = [p for p in extracted_info_from_papers if p['arxiv_id'] in paper_ids]
    
    group_papers_summary = []
    for paper in group_papers:
        paper_text = f"Title: {paper['title']}\nProblem: {paper['problem']}\nCompared baselines: {paper['compared_baselines']}\nResults: {paper['results']}\nLimitations: {paper['limitations']}\nNovelty: {paper['novelty']}\nkey_findings: {paper['key_findings']}\nopen_questions: {paper['open_questions']}"
        group_papers_summary.append(paper_text)
    group_papers_text = "\n\n".join(group_papers_summary)
    
    analysis = None
    msg_history = None
    
    for iteration in range(num_iterations):
        if iteration == 0:
            # Первичная генерация анализа
            response, msg_history = get_response_from_llm(
                msg=group_analysis_prompt.format(
                    group_name=group_name, 
                    topic=topic, 
                    papers_combined=group_papers_text
                ),
                print_debug=False,
                msg_history=msg_history,
                rate_limit=OPENROUTER_API_EXTRACT_RATE_LIMIT_SEC
            )
            analysis = response
        else:
            # Итеративная проверка и улучшение
            if analysis is None:
                logger.error(f'Первичная генерация анализа группы {group_name} вернула пустой текст\n')
                continue
            
            reflection_prompt = group_analysis_reflection_prompt.format(
                group_name=group_name,
                analysis=analysis,
                group_papers_text=group_papers_text                
            )
            
            response, msg_history = get_response_from_llm(
                msg=reflection_prompt,
                print_debug=False,
                msg_history=msg_history,
                rate_limit=OPENROUTER_API_EXTRACT_RATE_LIMIT_SEC
            )
            
            analysis = extract_response_block(response)

            # Если анализ удовлетворил, то извлекаем анализ из блока RESPONSE
            if "I am done" in response:
                break
    
    return analysis


def perform_groups_analysis(topic, papers, extracted_info_dir='extracted_info', output_dir='analysis_output'):
    """
    Выполняет группировку отобранных статей по группам со схожими областями исследования, и затем пишет анализ каждой из групп. 
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Загружаем извлеченную информацию
    papers_info = load_extracted_info(papers=papers, extracted_info_dir=extracted_info_dir)
    logger.info(f"Загружено {len(papers_info)} статей из {extracted_info_dir}\n")
    
    # Группируем статьи
    logger.info(f"Начинаю группировку статей на группы\n")
    groups = group_papers_by_subtopics(topic=topic, extracted_info_from_papers=papers_info)
    logger.info(f"По итогам этапа группировки статьи сгруппированы в {len(groups)} групп: {groups}\n")
    
    # Сохраняем распределение на группы в JSON
    with open(output_path / 'groups.json', 'w', encoding='utf-8') as f:
        json.dump(groups, f, indent=2, ensure_ascii=False)
    
    # Генерируем анализ для каждой группы
    group_analyses = {}
    for group_name, paper_ids in groups.items():
        safe_name = group_name.replace('/', '_').replace('\\', '_')
        analysis_file = output_path / f'group_{safe_name}.txt'
        
        if analysis_file.exists():
            logger.info(f"Анализ для группы '{group_name}' уже существует, загружаю из файла\n")
            with open(analysis_file, 'r', encoding='utf-8') as f:
                analysis = f.read()
        else:
            logger.info(f"Генерирую анализ для группы '{group_name}': ({len(paper_ids)} статей)\n")
            analysis = generate_group_analysis(papers_info, group_name, paper_ids, topic, num_iterations=3)
            # Сохраняем анализ группы
            with open(analysis_file, 'w', encoding='utf-8') as f:
                f.write(analysis)
        
        group_analyses[group_name] = analysis
    
    with open(output_path / 'group_analyses.json', 'w', encoding='utf-8') as f:
        json.dump(group_analyses, f, indent=2, ensure_ascii=False)
    
    return group_analyses


# Промпт для группировки статей по подтемам
group_papers_by_subtopics_prompt = """
Тебе дан список arxiv_id научных статей с извлеченной из текста статей ключевой информацией, связанной с темой: {topic}.
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
=== НАЧАЛО КЛЮЧЕВОЙ ИНФОРМАЦИИ ===
{papers_summary_text}
=== КОНЕЦ КЛЮЧЕВОЙ ИНФОРМАЦИИ ===

Отнеси каждую статью ТОЛЬКО К ОДНОЙ ПОДТЕМЕ. Отвечай только объектом JSON.
"""


# Промпт для исправления ошибок при группировке статей
improve_grouping_prompt = """
Текущая группировка статей:
=== НАЧАЛО ГРУППИРОВКИ ===
{groups}
=== КОНЕЦ ГРУППИРОВКИ ===

Проблемы: 
=== НАЧАЛО ПРОБЛЕМ ===
{feedback}
=== КОНЕЦ ПРОБЛЕМ ===

Исправь группировку: исправь все перечисленные проблемы. Каждая статья должна быть распределена ровно в одну самую подходящую группу без дубликатов. 
Ни одна статья не должна остаться вне группы. Суммарно групп должно быть 4 или меньше. Если групп получилось 5, то их число нужно сократить. Если статья не определна ни в одну группу, нужно определить ее в самую подходящую. 

Ответь ТОЛЬКО исправленным JSON.
"""


# Промпт для генерации анализа группы статей
group_analysis_prompt = """
Ты — агент научного синтеза, создающий промежуточный аналитический артефакт для последующей генерации обзорной научной статьи по теме {topic}.

Тебе дана группа научных статей {group_name}, объединенных одной подтемой.

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
=== НАЧАЛО ТЕКСТА СТАТЕЙ ===
{papers_combined}
=== КОНЕЦ ТЕКСТА СТАТЕЙ ===

Пиши анализ в структурированном формате с четкими заголовками, начиная каждый раздел с ##.
"""


# Объединённый промпт для проверки и улучшения анализа группы статей
group_analysis_reflection_prompt = """
Проверь анализ подтемы {group_name} на полноту, точность, согласованность, relevance к статьям, научную строгость и ясность структуры.
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

Если анализ по твоему хороший и изменения не требуются, напиши "I am done" под заголовком THOUGHT, а в RESPONSE верни ТОЧНО ТАКОЙ ЖЕ предыдущий текст анализа.
ВКЛЮЧАЙ "I am done" под заголовок THOUGHT ТОЛЬКО ЕСЛИ ТЫ НЕ ДЕЛАЕШЬ ИЗМЕНЕНИЙ. Отвечай строго по формату.

Текст текущего анализа:
=== НАЧАЛО ТЕКСТА АНАЛИЗА ===
{analysis}
=== КОНЕЦ ТЕКСТА АНАЛИЗА ===

Текст использованных статей:
=== НАЧАЛО ТЕКСТА СТАТЕЙ ===
{group_papers_text}
=== КОНЕЦ ТЕКСТА СТАТЕЙ ===
"""


# if __name__ == "__main__":
    # Тестирую генерацию анализов по группам
    # test_topic = 'multiagent systems of science automation'
    # from search_agent import get_set_number_of_papers
    # from extraction_agent import download_and_extract_arxiv_papers
    # selected_papers = get_set_number_of_papers(topic=test_topic, num_of_selected_papers=10, total_num_of_papers=70, num_of_expanded_queries=7, papers_per_query=10, store_path='logs/arxiv_search_log.json')
    # extracted_info_from_papers = download_and_extract_arxiv_papers(selected_papers, pdf_dir='pdfs', txt_dir='txts', extracted_info_dir='extracted_info')

    # group_analyses = perform_groups_analysis(topic=test_topic, papers=selected_papers, extracted_info_dir='extracted_info', output_dir='analysis_output')
