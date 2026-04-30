from arxiv_agent import search_and_download_arxiv_papers, extract_and_save_texts
from extraction_agent import extract_key_info_from_papers, perform_grouping_and_analysis


if __name__ == "__main__":
    user_topic = input('Введите тему для проведения исследования (тему укажите на английском): ').strip()

    if not user_topic:
        print('Тема не задана, завершение.')
    else:
        print(f'[DEBUG] Запуск исследования по теме: {user_topic}')

        # Обнуление файла с логами LLM
        with open('logs/llm_logs.txt', 'w', encoding='utf-8'):
            pass

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