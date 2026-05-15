from utils import load_extracted_info, load_logger, init_token_usage_tracking, save_token_usage
from search_agent import get_set_number_of_papers
from extraction_agent import download_and_extract_arxiv_papers
from analysis_agent import perform_groups_analysis
from writeup_agent import perform_review_writeup


logger = load_logger()

def write_review(topic, abstract, num_of_selected_papers=10, total_num_of_papers=70, num_of_expanded_queries=7, papers_per_query=10, store_path='logs/arxiv_search_log.json', local_papers_dir='diploma/input_papers', max_local_papers_num=10, pdf_dir='pdfs', txt_dir='txts', extracted_info_dir='extracted_info', analysis_dir='analysis_output'):
# Обнуление файла с логами
    with open('logs/app.log', 'w') as f:
        pass
    
    # Инициализируем трекер токенов
    init_token_usage_tracking()
    
    try:
        selected_papers = get_set_number_of_papers(topic=topic, abstract=abstract, num_of_selected_papers=num_of_selected_papers, total_num_of_papers=total_num_of_papers, num_of_expanded_queries=num_of_expanded_queries, papers_per_query=papers_per_query, store_path=store_path, local_papers_dir=local_papers_dir, max_local_papers_num=max_local_papers_num)

        extracted_info_paths = download_and_extract_arxiv_papers(selected_papers, pdf_dir=pdf_dir, txt_dir=txt_dir, extracted_info_dir=extracted_info_dir)

        group_analyses = perform_groups_analysis(topic=topic, papers=selected_papers, extracted_info_dir=extracted_info_dir, output_dir=analysis_dir)

        extracted_info_from_papers = load_extracted_info(selected_papers)
        perform_review_writeup(topic=topic, group_analyses=group_analyses, extracted_info_from_papers=extracted_info_from_papers, output_dir=analysis_dir)

        logger.info('Итоговый обзор сгенерирован. Завершаю работу\n')
        
        # Сохраняем информацию об использовании токенов
        save_token_usage(topic, output_dir=analysis_dir)
    
    except RuntimeError as e:
        logger.error(f'Ошибка search_agent: {e}. Завершаю работу.')
        return 


if __name__ == "__main__":
    user_topic = input('Введите тему для проведения исследования (тему укажите на английском): ').strip()
    user_abstract = input('Введите аннотацию для проведения исследования (при желании, иначе - оставьте пустым): ').strip()

    if not user_topic:
        logger.error('Тема не задана, завершение\n')
    else:
        logger.info(f'Запуск исследования по теме: {user_topic}\n')
        write_review(user_topic, user_abstract)

        
# темы:
# 
# multiagent systems of science automation
# personalized question generator for graduate project defense simulator

# Robustness of Large Language Models in Medical Question Answering
# agentic systems of science automation for searching and initial analysis of papers
# Approximation algorithms for the generalized travelling salesman problem