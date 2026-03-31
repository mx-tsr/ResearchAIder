import os
import requests
import time


''' To DO:
Идея, что сейчас нужно сделать:
1. Получить релевантную статью по заданной теме
2. По ее ID скачать ее и сохранить в папку
3. Извлечь из PDF текст и сохранить его в txt файл

Пока что с sematic Scolar API хрень какая-то, не могу понять, что не так. Надо попробовать с arXiv API. 

4. Собрать все тексты статей в txt файлы (не json?)
5. Подумать, как сформировать промпт, в который можно будет подавать много текста, возможно текст одной статьи - это много. 
Как сделать так, чтобы ллм не забывала контекст и делала итеративно лучше. 
6. Сделать промпт, который будет доставать ключевую информацию из статей и сохранять в json файл.
7. Мета-анализ + Формирование json с мыслями?
8. Сделать промпт, который будет анализировать мысли, делать мета-анализ и остальное, что я закладывал
9. Засовывания идей в промпт и генерация обзора.  

P.S. Насчет того, что писать в результатах. Возможно сравнивать не просто с ручным анализом, а с ручным анализом + обращение к нейронке.
Плюсом описать, почему агенты круче, чем такой подход (такие статьи есть). 
'''

def download_pdf(url, path, user_agent = 'requests/2.0.0'):
    # send a user-agent to avoid server error
    headers = {
        'user-agent': user_agent,
    }

    # stream the response to avoid downloading the entire file into memory
    response = requests.get(
        url, 
        headers=headers, 
        stream=True, 
    verify=False)

    response.raise_for_status()

    if response.headers['content-type'] != 'application/pdf':
        raise Exception('The response is not a pdf')

    with open(path, 'wb') as f:
        # write the response to the file, chunk_size bytes at a time
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)


def download_paper(paper, directory='papers', user_agent='requests/2.0.0'):

    if not paper.get('isOpenAccess'):
        return None
    
    open_access = paper.get('openAccessPdf')

    if not open_access or not open_access.get('url'):
        return None

    paperId = paper['paperId']
    pdf_url = open_access['url']
    pdf_path = os.path.join(directory, f'{paperId}.pdf')

    # create the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)

    # check if the pdf has already been downloaded
    if not os.path.exists(pdf_path):
        download_pdf(pdf_url, pdf_path, user_agent=user_agent)

    return pdf_path


def search_for_papers(query, result_limit=10, offset=0):
    if not query:
        return None
    response = requests.get(
        "https://api.semanticscholar.org/graph/v1/paper/search",
        headers={},
        params={
            "query": query,
            "limit": result_limit,
            "offset": offset,
            "fields": "paperId,title,abstract,isOpenAccess,openAccessPdf,externalIds"
        },
    )
    response.raise_for_status()
    results = response.json()
    time.sleep(2.0)
    papers = results["data"]
    return papers


def search_for_papers_with_retry(query, result_limit=10, offset=0, max_retries=3):
     for attempt in range(max_retries):
        try:
            return search_for_papers(query, result_limit, offset)
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(min(2 ** (attempt + 1), 60))  # Exponential backoff with a maximum wait time of 60 seconds
            print(f"Request failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying...")


query = 'science automation using agentic systems'
papers = search_for_papers_with_retry(query, result_limit=10, offset=20, max_retries=4)
print(papers)

for paper in papers:
    try:
        result = download_paper(paper)
    except Exception as e:
        print(f'Error downloading paper {paper["paperId"]}: {e}')
