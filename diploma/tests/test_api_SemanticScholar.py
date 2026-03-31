import os
from dotenv import load_dotenv
import requests
import json
import time

load_dotenv()

def search_for_papers(query, result_limit=1):
    if not query:
        return None

    s2_api_key = os.getenv("S2_API_KEY")
    headers = {"x-api-key": s2_api_key} if s2_api_key else {}

    response = requests.get(
        "https://api.semanticscholar.org/graph/v1/paper/search",
        headers=headers,
        params={
            "query": query,
            "limit": result_limit,
            "fields": "title,authors,venue,year,abstract,citationStyles,citationCount",
        },
    )
    print(f"Response Status Code: {response.status_code}")
    print(
        f"Response Content: {response.text[:500]}"
    )  # Print the first 500 characters of the response content
    response.raise_for_status()
    results = response.json()
    print(results)
    total = results["total"]
    time.sleep(1.0)
    if not total:
        return None
    
    papers = results["data"]
    return papers


query = '"science automation using agentic systems"'
response = search_for_papers(query)
print(response)

