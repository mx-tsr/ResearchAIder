from dotenv import load_dotenv
from arxiv_agent import search_and_download_arxiv_papers, extract_and_save_texts
from extraction_agent import extract_key_info_from_papers, perform_grouping_and_analysis

load_dotenv()
''' TODO:
Добавить индикатор того, что просиходит скачивание статей или другой процесс через tqdm
'''

'''
Пайплайн:
1. Получаю от пользователя тему исследования / название научной статьи / ее pdf
2. Семантический поиск в arxiv и скачивание статей
  а. Генерирую расширенный набор запросов на основе темы
  б. Ищу статьи по расширенному набору запросов
  в. Среди всех найденных статей отбираю 10 самых релевантных
  г. Скачиваю их pdf
  д. Кэширую запросы к arxiv, обновляю раз в 24 часа
3. Извлечение текста из pdf, сохранение только полезной части и сохранение в txt файлы





5. Подумать, как сформировать промпт, в который можно будет подавать много текста, возможно текст одной статьи - это много. 
Как сделать так, чтобы ллм не забывала контекст и делала итеративно лучше. 
6. Сделать промпт, который будет доставать ключевую информацию из статей и сохранять в json файл.
7. Мета-анализ + Формирование json с мыслями?
8. Сделать промпт, который будет анализировать мысли, делать мета-анализ и остальное, что я закладывал
9. Засовывания идей в промпт и генерация обзора.  

P.S. Насчет того, что писать в результатах. Возможно сравнивать не просто с ручным анализом, а с ручным анализом + обращение к нейронке.
Плюсом описать, почему агенты круче, чем такой подход (такие статьи есть). 
'''


if __name__ == "__main__":
    user_topic = input('Введите тему для проведения исследования (тему укажите на английском): ').strip()
    if not user_topic:
        print('Тема не задана, завершение.')
    else:
        print(f'[DEBUG] Запуск исследования по теме: {user_topic}')

        search_and_download_output = search_and_download_arxiv_papers(user_topic, num_of_selected_papers=10, total_num_of_papers=70, num_of_expanded_queries=5, store_results='logs/arxiv_search_log.json')
        selected_papers = search_and_download_output["selected_papers"]

        print(f'[DEBUG] Найдено {len(selected_papers)} статей, сохранено: {len(search_and_download_output["pdf_paths"])} PDF файлов')
        print(f'[DEBUG] Лог: {search_and_download_output["log_file"]}')

        # Извлечение текста из PDF и сохранение в txt
        extract_and_save_texts(selected_papers, txt_dir='txts', pdf_dir='pdfs')
        print('[DEBUG] Текст извлечен из PDF и сохранен в txt файлы')

        # Извлечение ключевой информации
        extract_key_info_from_papers(selected_papers, output_dir='extracted_info', txt_dir='txts')
        print('[DEBUG] Ключевая информация извлечена и сохранена в JSON')
        extract_and_save_texts(selected_papers, txt_dir='txts', pdf_dir='pdfs')
        print('[DEBUG] Текст извлечен из PDF и сохранен в txt файлы')

        # Извлечение ключевой информации
        extract_key_info_from_papers(selected_papers, txt_dir='txts', output_dir='extracted_info')
        print('[DEBUG] Ключевая информация извлечена и сохранена в JSON')

        perform_grouping_and_analysis(user_topic, selected_papers, extracted_dir='extracted_info', output_dir='analysis_output')
        print('[DEBUG] Группировка и анализ завершены')