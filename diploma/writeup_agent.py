from pathlib import Path

from llm import get_response_from_llm
from utils import load_logger, extract_response_block

OPENROUTER_API_EXTRACT_RATE_LIMIT_SEC = 5.0 


logger = load_logger()


def extract_relevant_parts(analyses, relevant_headers):
    '''
    Извлекает релевантные части из анализов по заголовкам (маркеры ##).
    Использует структурированный формат с явными маркерами разделов.
    '''
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
                        result_text.append(f"Из группы '{group_name}' ({section_name}):\n{content}\n")
    
    return '\n\n'.join(result_text) if result_text else f"[Не найдено релевантных разделов для: {relevant_headers}]"


def generate_final_review(topic, group_analyses, num_iterations=3):
    '''
    Генерирует итоговый обзор статьи по фиксированным разделам.
    '''
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
    
    # Какие части анализа групп подходят для генерации раздела итогового обзора
    section_mapping = {
        "Введение": ["Обзор подтемы"],
        "Предметная область": ["Обзор подтемы", "Основные идеи и методы"],
        "Методы и Подходы": ["Основные идеи и методы", "Сильные стороны и ограничения"],
        "Ключевые Результаты и Открытия": ["Совпадающие выводы и подтверждающие результаты", "Различия и противоречия"],
        "Ограничения и Проблемы": ["Сильные стороны и ограничения", "Различия и противоречия"],
        "Будущие Направления": ["Пробелы исследований и открытые вопросы"],
        "Заключение": ["Ключевые выводы для итогового мета-анализа"]
    }
    
    final_sections = {}
    
    for section_name in sections:
        logger.info(f"Генерирую раздел: {section_name}\n")
        relevant_parts = extract_relevant_parts(group_analyses, section_mapping[section_name])
        
        section_content = None
        msg_history = None
        
        for iteration in range(num_iterations):
            if iteration == 0 or section_content is None:
                # Первичная генерация раздела без ссылок (или повтор, если предыдущая не удалась)
                response, msg_history = get_response_from_llm(
                    msg=generate_section_prompt.format(
                        section_name=section_name,
                        topic=topic,
                        relevant_parts=relevant_parts
                    ),
                    print_debug=False,
                    msg_history=None,  # Сброс истории для первичной генерации
                    rate_limit=OPENROUTER_API_EXTRACT_RATE_LIMIT_SEC
                )
                section_content = response
            else:
                # Итеративная проверка и улучшение
                # Ограничиваем историю до последних 2 сообщений, чтобы избежать переполнения контекста
                limited_msg_history = msg_history[-2:] if msg_history and len(msg_history) > 2 else msg_history
                response, msg_history = get_response_from_llm(
                    msg=review_section_reflection_prompt.format(
                        section_name=section_name,
                        topic=topic,
                        section_content=section_content,
                        relevant_parts=relevant_parts
                    ),
                    print_debug=False,
                    msg_history=limited_msg_history,
                    rate_limit=OPENROUTER_API_EXTRACT_RATE_LIMIT_SEC
                )
                
                new_section_content = extract_response_block(response)
                if new_section_content is not None:
                    section_content = new_section_content
                    # Если раздел удовлетворил, то извлекаем его из блока RESPONSE
                    if "I am done" in response:
                        break
                else:
                    # Если не удалось извлечь улучшенный текст, используем текущий и останавливаемся
                    break
        
        logger.info(f"Раздел {section_name} сгенерирован: \n{section_content}\n")
        if section_content is None:
            logger.error(f"Раздел {section_name} не сгенерирован, пропускаю\n")
            continue
        final_sections[section_name] = section_content
    
    # Собираем итоговую статью без ссылок
    full_review = "\n\n".join([f"# {section_name}\n\n{content}" for section_name, content in final_sections.items()])
    
    logger.info('Полный обзор без ссылок сформирован. Начинаю проверку всей статьи\n')
    
    # Проверка всей статьи
    msg_history = None
    response, msg_history = get_response_from_llm(
        msg=final_review_check_prompt.format(
            topic=topic,
            full_review=full_review
        ),
        print_debug=False,
        msg_history=msg_history,
        rate_limit=OPENROUTER_API_EXTRACT_RATE_LIMIT_SEC
    )
    
    new_full_review = extract_response_block(response)
    if new_full_review is not None:
        full_review = new_full_review
    # Если не удалось извлечь улучшенный текст, используем текущий full_review
    
    return full_review  


def split_review_into_sections(review_text):
    sections = []
    current_header = None
    current_lines = []

    for line in review_text.splitlines():
        if line.startswith('# '):
            if current_header is not None:
                sections.append((current_header, '\n'.join(current_lines).strip()))
            current_header = line.strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_header is not None:
        sections.append((current_header, '\n'.join(current_lines).strip()))
    elif review_text.strip():
        sections.append(('# Review', review_text.strip()))

    return sections


def add_citations_to_section(topic, section_header, section_body, papers_info_text, num_iterations=3):
    section_text = f"{section_header}\n\n{section_body.strip()}"
    msg_history = None

    for iteration in range(num_iterations):
        if iteration == 0 or not (section_text.startswith(section_header) and len(section_text) > len(section_header)):
            # Добавляем ссылки в секцию (или повтор, если предыдущая не удалась)
            response, msg_history = get_response_from_llm(
                msg=section_citation_prompt.format(
                    topic=topic,
                    section_text=section_text,
                    papers_info=papers_info_text,
                ),
                print_debug=False,
                msg_history=msg_history,
                temperature=0.3,
                rate_limit=OPENROUTER_API_EXTRACT_RATE_LIMIT_SEC,
            )
            section_text_with_citations = extract_response_block(response, label='RESPONSE')

            if section_text_with_citations and section_header in section_text_with_citations:
                section_text = section_text_with_citations
        else:
            # Проверяем и улучшаем добавление ссылок
            check_response, msg_history = get_response_from_llm(
                msg=section_citation_reflection_prompt.format(
                    section_text=section_text,
                    papers_info_text=papers_info_text
                ),
                print_debug=False,
                msg_history=msg_history,
                rate_limit=OPENROUTER_API_EXTRACT_RATE_LIMIT_SEC,
            )
            check_section_text_with_citations= extract_response_block(check_response, label='RESPONSE')

            if check_section_text_with_citations and section_header in check_section_text_with_citations:
                section_text = check_section_text_with_citations
                if "I am done" in check_response:
                    break
            else:
                # Если не удалось извлечь улучшенный текст, используем текущий и останавливаемся
                break

    return section_text


def add_citations_to_review(topic, review_text, extracted_info_from_papers, num_iterations=3):
    """
    Добавляет ссылки на статьи в готовый обзор по разделам
    """
    papers_list = []
    for p in extracted_info_from_papers:
        papers_list.append(
            f"- Arxiv_id: {p['arxiv_id']}\n Title: {p['title']}\n Problem: {p['problem']}\n Results: {p['results']}"
        )
    papers_info_text = "\n".join(papers_list)

    sections = split_review_into_sections(review_text)
    updated_sections = []

    for section_header, section_body in sections:
        updated_section = add_citations_to_section(topic=topic, section_header=section_header, section_body=section_body, papers_info_text=papers_info_text, num_iterations=num_iterations)
        updated_sections.append(updated_section)

    review_with_citations = "\n\n".join(updated_sections)

    required_headers = [
        "# Введение",
        "# Предметная область",
        "# Методы и Подходы",
        "# Ключевые Результаты и Открытия",
        "# Ограничения и Проблемы",
        "# Будущие Направления",
        "# Заключение",
    ]
    if not all(header in review_with_citations for header in required_headers):
        logger.info("Итоговый обзор со ссылками потерял структуру, возвращаю исходный обзор без изменений\n")
        return review_text

    return review_with_citations


def generate_hypotheses_section(topic, extracted_info_from_papers):
    """
    Генерирует раздел гипотез на основе имеющихся анализов и использованных статей.
    """
    paper_summaries = []
    for p in extracted_info_from_papers:
        paper_summaries.append(
            f"- Arxiv_id: {p['arxiv_id']}\n Title: {p['title']}\n Problem: {p.get('problem', '')}\n Results: {p.get('results', '')}\n Open_questions: {p.get('open_questions', '')}\n Implicit_gaps: {p.get('implicit_gaps', '')}"
        )
    papers_info_text = "\n".join(paper_summaries)

    logger.info("Начинаю писать раздел гипотезы\n")
    
    hypotheses_text, _ = get_response_from_llm(
        msg=hypotheses_prompt.format(
            topic=topic,
            final_review=papers_info_text,
        ),
        print_debug=False,
        temperature=0.9,
        rate_limit=OPENROUTER_API_EXTRACT_RATE_LIMIT_SEC,
    )
     
    logger.info("Изначальный текст раздела гипотезы создан. Начинаю его оценку\n")

    critique_text, _ = get_response_from_llm(
        msg=hypotheses_critic_prompt.format(
            hypotheses=hypotheses_text
        ),
        print_debug=False,
        temperature=0.3,
        rate_limit=OPENROUTER_API_EXTRACT_RATE_LIMIT_SEC,
    )

    logger.info("Оценка раздела гипотезы сделана. Начинаю улучшение\n")
    
    rewrite_response, _ = get_response_from_llm(
        hypotheses_rewrite_prompt.format(
            hypotheses=hypotheses_text,
            critique=critique_text
        ),
        msg_history=None,
        rate_limit=OPENROUTER_API_EXTRACT_RATE_LIMIT_SEC
    )

    improved_hypotheses_text = extract_response_block(rewrite_response, label='RESPONSE')
    if improved_hypotheses_text:
        hypotheses_text = improved_hypotheses_text
    # Если не удалось извлечь, используем текущий hypotheses_text

    return hypotheses_text


def perform_review_writeup(topic, group_analyses, extracted_info_from_papers, output_dir='analysis_output'):
    """
    Выполняет генерацию итогового обзора, добавление ссылок на использованные статьи и генерацию гипотез дальнейшего исследования
    """
    output_path = Path(output_dir)

    # Генерируем итоговый обзор без ссылок
    logger.info("Генерирую итоговый обзор без ссылок\n")
    final_review_no_citations = generate_final_review(topic=topic, group_analyses=group_analyses, num_iterations=3)
    
    # Сохраняем обзор без ссылок
    with open(output_path / 'final_review_no_citations.txt', 'w', encoding='utf-8') as f:
        f.write(final_review_no_citations)
    
    # Добавляем ссылки на статьи
    logger.info("Добавляю ссылки на статьи в обзор\n")
    final_review_with_citations = add_citations_to_review(topic=topic, review_text=final_review_no_citations, extracted_info_from_papers=extracted_info_from_papers, num_iterations=3)
    
    # Сохраняем итоговый обзор со ссылками
    with open(output_path / 'final_review_with_citations.txt', 'w', encoding='utf-8') as f:
        f.write(final_review_with_citations)
    
    # Генерируем раздел гипотез и дальнейших исследований 
    logger.info("Генерирую раздел гипотез и дальнейших исследований\n")
    hypotheses_text = generate_hypotheses_section(topic=topic, extracted_info_from_papers=extracted_info_from_papers)
    review = final_review_with_citations.strip() + "\n\n# Гипотезы и дальнейшие исследования\n\n" + hypotheses_text

    # Сохраняем обзор уже с разделом гипотез
    with open(output_path / 'final_review.txt', 'w', encoding='utf-8') as f:
        f.write(review)
    
    logger.info(f"Генерация обзора завершена. Результаты сохранены в {output_dir}\n")
    return review


# Промпт для генерации раздела итогового обзора
generate_section_prompt = """
Напиши раздел "{section_name}" научной обзорной статьи по теме "{topic}".

Используй эту релевантную информацию из анализов групп:
=== НАЧАЛО ИНФОРМАЦИИ ===
{relevant_parts}
=== КОНЕЦ ИНФОРМАЦИИ ===

Требования:
- Научный стиль, синтез (не пересказ).
- Сравнивай подходы между группами.
- НЕ добавляй ссылки на статьи - они будут добавлены отдельно.
- Учитывай противоречия и консенсус.
- Раздел уровня обзорной публикации.
- Пиши ТОЛЬКО ЭТОТ РАЗДЕЛ, БЕЗ ВЫВОДОВ К НЕМУ.
"""


# Промпт для написания и итеративного улучшения раздела обзора
review_section_reflection_prompt = """
СТРОГО СОБЛЮДАЙ ФОРМАТ ОТВЕТА:

THOUGHT:
<твои размышления о разделе>

RESPONSE:
```
<улучшенный раздел или старый текст раздела, если текст идеален>
```

Проверь и улучши раздел "{section_name}" обзорной статьи по теме "{topic}".

Оцени и исправь:
- полноту информации из предоставленных материалов
- согласованность и логичность изложения
- научную строгость и точность
- глубину обработки материала
- отсутствие важной информации из анализов групп
- повторы информации, пропущенные переходы в рассуждениях, краткость без потери смысла

Всегда предлагай улучшения. Даже если текст кажется хорошим, предложи хотя бы одно мелкое улучшение (например, перефразировку для ясности или структуры).
Скажи "I am done" в THOUGHT ТОЛЬКО если текст идеален и не требует НИКАКИХ изменений, даже мелких. В этом случае верни ТОЧНО ТАКОЙ ЖЕ предыдущий текст раздела в RESPONSE.

ВАЖНО: Пиши в этом промпте ТОЛЬКО ТОТ РАЗДЕЛ, БЕЗ ВЫВОДОВ И ЗАКЛЮЧЕНИЙ К НЕМУ.

Текущий раздел:
=== НАЧАЛО ТЕКУЩЕГО РАЗДЕЛА ===
{section_content}
=== КОНЕЦ ТЕКУЩЕГО РАЗДЕЛА ===

Релевантная информация:
=== НАЧАЛО ИНФОРМАЦИИ ===
{relevant_parts}
=== КОНЕЦ ИНФОРМАЦИИ ===

НАПОМИНАНИЕ: ОБЯЗАТЕЛЬНО пиши THOUGHT и RESPONSE в указанном формате. НЕЛЬЗЯ давать ответ без RESPONSE.
"""

# Промпт для проверки всего обзора (после написания без ссылок)
final_review_check_prompt = """
СТРОГО СОБЛЮДАЙ ФОРМАТ ОТВЕТА:

THOUGHT:
<твои размышления о статье>

RESPONSE:
```
<улучшенная статья или старый текст статьи, если текст идеален>
```

Проверь и улучши целиком обзорную статью по теме "{topic}" на:
- согласованность разделов
- повторы в информации
- пропущенные переходы в рассуждениях
- противоречия
- недостающий синтез
- полноту раздела с пробелами исследований
- краткость без потери смысла
- научную точность и логичность

Всегда предлагай улучшения. Даже если статья кажется хорошей, предложи хотя бы одно мелкое улучшение (например, перефразировку переходов или уточнение), пиши текст улучшенной статьи в RESPONSE.
Скажи "I am done" в THOUGHT ТОЛЬКО если статья идеальна и не требует НИКАКИХ изменений, даже мелких. В этом случае верни ТОЧНО ТАКОЙ ЖЕ предыдущий текст статьи в RESPONSE.

НАПОМИНАНИЕ: ОБЯЗАТЕЛЬНО пиши THOUGHT и RESPONSE в указанном формате.

Полный текст обзора:
=== НАЧАЛО ТЕКСТА ОБЗОРА ===
{full_review}
=== КОНЕЦ ТЕКСТА ОБЗОРА ===
"""


# Промпт для добавления ссылок в одну секцию обзора
section_citation_prompt = """
Ты — агент, добавляющий ссылки на статьи в один раздел обзорной статьи по теме "{topic}".

Раздел:
=== НАЧАЛО ТЕКСТА РАЗДЕЛА ДАННЫХ ===
{section_text}
=== КОНЕЦ ТЕКСТА РАЗДЕЛА ДАННЫХ ===

Ниже справочные данные о статьях. Это контекст для анализа, НЕ копируй этот текст дословно в обзор. Используй его только как источник фактов.

=== НАЧАЛО СПРАВОЧНЫХ ДАННЫХ ===
{papers_info}
=== КОНЕЦ СПРАВОЧНЫХ ДАННЫХ ===

Задача: добавь ссылки [CITATION: arxiv_id | название статьи] только в тех местах, где это нужно для подтверждения важных утверждений или прямых ссылок на работы.

ВАЖНО:
- Не меняй смысл и структуру раздела.
- Используй ТОЧНЫЕ arxiv_id и названия.
- Не добавляй новые идеи.
- Не переписывай предложения.
- Сохрани заголовок раздела в начале.
- Если ссылки не нужны, оставь раздел без изменений. 
- Писать ТОЛЬКО на русском, кроме названий статей, на которые ссылаешься и терминов. Предложения на английском запрещены. 

Отвечай строго в формате:

THOUGHT:
<что сделано>

RESPONSE:
```
<тот же раздел с добавленными ссылками>
```

НЕ ПИШИ ссылки там, где нет прямых отсылок к работам, НЕ вставляй бездумно ссылки. НЕ пиши содержание из списка доступных статей.
"""


# Промпт для улучшения добавления ссылок в один раздел обзора
section_citation_reflection_prompt = """
Проверь этот раздел на наличие всех необходимых ссылок. Не меняй текст, кроме добавления ссылок. Если ссылки добавлены полностью, ответь в RESPONSE точно тем же текстом секции и добавь в THOUGHT 'I am done'. Если нужно добавить ещё ссылки, верни обновлённую секцию и НЕ пиши 'I am done'. Не пиши ничего кроме блоков THOUGHT и RESPONSE (никаких примечаний).
ПИШИ весь текст на русском, кроме названий статей, на которые ссылаешься. 

Отвечай строго по шаблону:
THOUGHT:
< подтверждающие мысли >

RESPONSE:
```
< старый текст секции или текст с дополненными ссылками >
```

Текущая версия раздела со ссылками:
=== НАЧАЛО РАЗДЕЛА ===
{section_text}
=== КОНЕЦ РАЗДЕЛА ===

Доступные статьи (используй ТОЧНЫЕ arxiv_id и названия):
=== НАЧАЛО ДОСТУПНЫХ СТАТЕЙ ===
{papers_info_text}
=== КОНЕЦ ДОСТУПНЫХ СТАТЕЙ ===
"""


# Промпт для генерации раздела гипотез и дальнейших исследований
hypotheses_prompt = '''
На основе обзорной статьи по теме {topic} тебе нужно написать в нее раздел "Гипотезы и дальнейшее исследование". 
В разделе нужно предложить 1-2 гипотезы, связанные с темой, которые будут новыми, ранее не исследованными и не рассмотренными. 
Гипотеза должна включать в себя конкретный вопрос для исследования, быть сформулирована как название статьи и иметь описание, что в ней предлагается сделать и в чем ее новизна. 
Например, в теме "2d diffusion" гипотеза могла бы быть "Diffusion Experts: A Single Model with Multiple Specialists for Improved Mode Coverage" с описанием "Implement a single diffusion model with multiple 'experts' that specialize in different aspects of the data distribution. Use a gating mechanism to select the most relevant expert for each input. Evaluate performance using estimated KL divergence, mode coverage, and visual inspection."

Требования:
- Название каждой гипотезы должно быть сформулировано в виде названия научной статьи.
- Гипотезы НЕ должны существенно пересекаться с существующей литературой или быть уже хорошо изученными.
- Раздел назови "Гипотезы для дальнейшего исследования", напиши почему каждая гипотеза является новой и не покрыта в использованной литературе.
- Не генерируй шаблонные идеи, фокусируйся на слепых зонах, выявленных из предоставленных статей.

Пиши только текст раздела. 

Текст итогового обзора: 
=== НАЧАЛО ТЕКСТА ИТОГОВОГО ОБЗОРА ===
{final_review}
=== КОНЕЦ ТЕКСТА ИТОГОВОГО РАЗБОРА ===
'''


hypotheses_critic_prompt = '''
Проанализируй предложенные гипотезы как научный критик. Будь строг к их новизне. 

Раздел "Гипотезы для дальнейшего исследования":

=== НАЧАЛО ТЕКСТА РАЗДЕЛА ===
{hypotheses}
=== КОНЕЦ ТЕКСТА РАЗДЕЛА ===

Проверь:

1. Есть ли банальные или очевидные идеи
2. Есть ли идеи, уже фактически присутствующие в обзоре или общеизвестны
3. Какие гипотезы недостаточно новы
4. Какие гипотезы слишком общие
5. Как усилить новизну и экспериментальную проверяемость

Дай конструктивный критический отзыв.

Пиши только критику и рекомендации по улучшению.
'''


hypotheses_rewrite_prompt = '''
Ниже привиден исходный текст раздела "Гипотезы для дальнейшего исследования":

=== НАЧАЛО ТЕКСТА РАЗДЕЛА ===
{hypotheses}
=== КОНЕЦ ТЕКСТА РАЗДЕЛА ===

Ниже критические замечания к этому разделу:

=== НАЧАЛО ТЕКСТА ЗАМЕЧАНИЙ ===
{critique}
=== КОНЕЦ ТЕКСТА ЗАМЕЧАНИЙ ===

Перепиши раздел "Гипотезы для дальнейшего исследования", улучшив:
- новизну
- конкретность
- научную ценность
- экспериментальную проверяемость
- исправив критические замечания

Если гипотеза хорошая — сохрани её. Если слабая — переработай. НИЧЕГО НЕ пиши, кроме улучшенного текста раздела, без выводов и комментариев, только текст раздела.

Ответ СТРОГО в формате:

THOUGHT:
<краткое объяснение изменений>

RESPONSE:
```
<улучшенный финальный раздел гипотез>
```
'''


if __name__ == "__main__":
    test_topic = 'multiagent systems of science automation'
    from utils import load_extracted_info
    from search_agent import get_set_number_of_papers
    import json
    selected_papers = get_set_number_of_papers(topic=test_topic, num_of_selected_papers=10, total_num_of_papers=70, num_of_expanded_queries=7, papers_per_query=10, store_path='logs/arxiv_search_log.json')
    extracted_info_from_papers = load_extracted_info(selected_papers)
    with open('analysis_output/group_analyses.json', 'r', encoding='utf-8') as f:
        group_analyses = json.load(f)
    perform_review_writeup(topic=test_topic, group_analyses=group_analyses, extracted_info_from_papers=extracted_info_from_papers, output_dir='analysis_output')