from utils import load_extracted_info, load_logger
from search_agent import get_set_number_of_papers
from extraction_agent import download_and_extract_arxiv_papers
from analysis_agent import perform_groups_analysis
from writeup_agent import perform_review_writeup


logger = load_logger()


if __name__ == "__main__":
    user_topic = input('Введите тему для проведения исследования (тему укажите на английском): ').strip()

    if not user_topic:
        logger.error('Тема не задана, завершение\n')
    else:
        logger.info(f'Запуск исследования по теме: {user_topic}\n')

        # Обнуление файла с логами
        with open('logs/app.log', 'w') as file:
            pass
        
        selected_papers = get_set_number_of_papers(topic=user_topic, num_of_selected_papers=10, total_num_of_papers=70, num_of_expanded_queries=7, papers_per_query=10, store_path='logs/arxiv_search_log.json')

        extracted_info_from_papers = download_and_extract_arxiv_papers(selected_papers, pdf_dir='pdfs', txt_dir='txts', extracted_info_dir='extracted_info')

        group_analyses = perform_groups_analysis(topic=user_topic, papers=selected_papers, extracted_info_dir='extracted_info', output_dir='analysis_output')

        extracted_info_from_papers = load_extracted_info(selected_papers)
        perform_review_writeup(topic=user_topic, group_analyses=group_analyses, extracted_info_from_papers=extracted_info_from_papers, output_dir='analysis_output')

        logger.info('Итоговый обзор сгенерирован. Завершаю работу\n')

# темы:
# automated c
# multiagent systems of science automation
# agentic systems of science automation for searching and initial analysis of papers