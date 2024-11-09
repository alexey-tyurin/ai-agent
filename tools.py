import time
import json
import re
import asyncio
from random import uniform
from typing import List, Dict, TypedDict
import requests
import os
import getpass
from bs4 import BeautifulSoup
from langchain_core.tools import tool
from duckduckgo_search import DDGS
from duckduckgo_search.exceptions import RatelimitException
from langchain_openai import ChatOpenAI

import model

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

# Set API keys for OpenAI
_set_env("OPENAI_API_KEY")

# Initialize ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def run_async(coro):
    """Helper function to run async code in sync context"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)

async def async_invoke(llm: ChatOpenAI, prompt: str) -> str:
    """Asynchronously invoke the LLM"""
    response = await llm.ainvoke(prompt)
    return response.content

def normalize_arxiv_url(url):
    # Replace both pdf and html versions with abstract version
    normalized_url = re.sub(r'arxiv\.org/(pdf|html)/', 'arxiv.org/abs/', url)
    return normalized_url

@tool
def search_arxiv(inputs: str) -> List[model.Paper]:
    """Search arXiv papers using DuckDuckGo

    Args:
        inputs: JSON string containing search_query and max_results
    """

    # Parse the input JSON string
    input_data = json.loads(inputs)
    search_query = input_data['search_query']
    max_results = input_data.get('max_results', 20)  # Default to 20 if not specified

    max_retries = 3
    papers = []

    for attempt in range(max_retries):
        try:
            # Create new session for each attempt
            with DDGS() as ddgs:
                results = list(ddgs.text(search_query, max_results=max_results))
        except RatelimitException as e:
            if attempt == max_retries - 1:
                raise e
            sleep_time = uniform(5, 10)  # Longer delay between retries
            print(f"Rate limit hit, waiting {sleep_time:.2f} seconds before retry {attempt + 1}")
            time.sleep(sleep_time)
            continue
        except Exception as e:
            print(f"Unexpected error: {e}")
            raise

        for r in results:
            if "arxiv.org" in r["href"]:
                papers.append(model.Paper(
                    title=r["title"],
                    link=normalize_arxiv_url(r["href"]),
                    snippet=""
                ))
        for p in papers:
            p.snippet = get_arxiv_paper_abstract(p.link)

        return papers

def get_arxiv_paper_abstract(url):
    try:
        # Send request with headers to mimic a browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        # Raise an exception for bad status codes
        response.raise_for_status()

        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        # Extract abstract from paper
        return soup.find('blockquote', class_='abstract mathjax').text.replace('Abstract:', '').strip()

    except requests.RequestException as e:
        print(f"Error fetching the URL: {e}")
        return None
    except Exception as e:
        print(f"Error parsing the content: {e}")
        return None

@tool
def summarize_papers(text: str) -> List[model.PaperSummary]:
    """Summarize each paper using ChatOpenAI"""
    papers = json.loads(text)  # Convert string back to list of papers
    prompt_template = """You are a research assistant. Summarize the following paper in 3-4 sentences with maximum 100 words:
    
    {paper_text}
    
    Summary:"""

    async def process_papers():
        # Create list of coroutines
        tasks = [
            async_invoke(
                llm,
                prompt_template.format(paper_text=paper['snippet'])
            )
            for paper in papers
        ]

        # Execute all tasks concurrently
        summaries_content = await asyncio.gather(*tasks)

        # Create PaperSummary objects
        return [
            model.PaperSummary(
                title=paper['title'],
                link=paper['link'],
                summary=summary
            )
            for paper, summary in zip(papers, summaries_content)
        ]

    return run_async(process_papers())

@tool
def explain_relevance(data: str) -> List[model.PaperExplanation]:
    """Explain how each paper relates to the search intent using ChatOpenAI"""
    data_dict = json.loads(data)
    summaries = data_dict['summaries']
    intent = data_dict['intent']

    prompt_template = """Explain in 3-4 sentences with maximum 100 words how this paper could help with the following intent: {intent}

    Paper summary: {summary}
    
    Explanation:"""

    async def process_explanations():
        # Create list of coroutines
        tasks = [
            async_invoke(
                llm,
                prompt_template.format(
                    intent=intent,
                    summary=summary['summary']
                )
            )
            for summary in summaries
        ]

        # Execute all tasks concurrently
        explanations_content = await asyncio.gather(*tasks)

        # Create PaperExplanation objects
        return [
            model.PaperExplanation(
                title=summary['title'],
                link=summary['link'],
                summary=summary['summary'],
                explanation=explanation
            )
            for summary, explanation in zip(summaries, explanations_content)
        ]

    return run_async(process_explanations())

@tool
def rank_results(data: str) -> List[model.RankedPaper]:
    """Rank the papers based on relevance to intent using ChatOpenAI"""
    data_dict = json.loads(data)
    explanations = data_dict['explanations']
    intent = data_dict['intent']

    prompt_template = """Rate how well this paper matches the following intent on a scale of 0-10, where 10 is perfect match.
    Provide only the numerical score.
    
    Intent: {intent}
    Summary: {summary}
    Explanation: {explanation}
    
    Score:"""

    async def process_rankings():
        # Create list of coroutines
        tasks = [
            async_invoke(
                llm,
                prompt_template.format(
                    intent=intent,
                    summary=exp['summary'],
                    explanation=exp['explanation']
                )
            )
            for exp in explanations
        ]

        # Execute all tasks concurrently
        scores_content = await asyncio.gather(*tasks)

        # Process scores and create RankedPaper objects
        rankings = []
        for exp, score_str in zip(explanations, scores_content):
            try:
                score = float(score_str)
                score = max(0, min(10, score))  # Ensure score is between 0 and 10
            except:
                score = 0

            rankings.append(model.RankedPaper(
                title=exp['title'],
                link=exp['link'],
                summary=exp['summary'],
                explanation=exp['explanation'],
                score=score
            ))

        return sorted(rankings, key=lambda x: x.score, reverse=True)

    return run_async(process_rankings())
