from typing import List, Dict, Optional
from datetime import datetime
from pydantic import BaseModel, Field, validator
from duckduckgo_search import DDGS
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import GPT4All
from langgraph.graph import StateGraph, END
from langgraph.prebuilt.tool_executor import ToolExecutor
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
import gradio as gr
import uuid
import json
import requests
from bs4 import BeautifulSoup
import re

# Initialize MemorySaver
memory_saver = MemorySaver()

# Pydantic Models
class Paper(BaseModel):
    title: str
    link: str
    snippet: str

class PaperSummary(BaseModel):
    title: str
    link: str
    summary: str

class PaperExplanation(BaseModel):
    title: str
    link: str
    summary: str
    explanation: str

class RankedPaper(BaseModel):
    title: str
    link: str
    summary: str
    explanation: str
    score: float = Field(ge=0, le=10)

class SearchQuery(BaseModel):
    keywords: str
    intent: str
    from_date: str
    to_date: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    k: int = Field(default=10, ge=1, le=20)
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    @validator('from_date', 'to_date')
    def validate_date(cls, v):
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Date must be in YYYY-MM-DD format")
        return v

class AgentState(BaseModel):
    messages: List[Dict]
    papers: List[Paper] = Field(default_factory=list)
    summaries: List[PaperSummary] = Field(default_factory=list)
    explanations: List[PaperExplanation] = Field(default_factory=list)
    rankings: List[RankedPaper] = Field(default_factory=list)
    next_step: str
    session_id: str

    class Config:
        arbitrary_types_allowed = True

    def dict(self, *args, **kwargs):
        """Custom dict method to handle Pydantic models in lists"""
        d = super().dict(*args, **kwargs)
        d['papers'] = [paper.dict() for paper in self.papers]
        d['summaries'] = [summary.dict() for summary in self.summaries]
        d['explanations'] = [exp.dict() for exp in self.explanations]
        d['rankings'] = [rank.dict() for rank in self.rankings]
        return d

def normalize_arxiv_url(url):
    # Replace both pdf and html versions with abstract version
    normalized_url = re.sub(r'arxiv\.org/(pdf|html)/', 'arxiv.org/abs/', url)

    # Remove .pdf extension if present
    # normalized_url = re.sub(r'\.pdf$', '', normalized_url)

    return normalized_url

# Tools
# @tool
def search_arxiv(keywords: str, from_date: str, to_date: str, max_results: int) -> List[Paper]:
    """Search arXiv papers using DuckDuckGo"""
    with DDGS() as ddgs:
        search_query = f"site:arxiv.org {keywords} {from_date}..{to_date}"
        results = list(ddgs.text(search_query, max_results=max_results))
        papers = []
        for r in results:
            if "arxiv.org" in r["href"]:
                papers.append(Paper(
                    title = r["title"],
                    link = normalize_arxiv_url(r["href"]),
                    snippet = ""
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

# Graph functions
def search_papers(state: AgentState) -> AgentState:
    messages = state.messages
    query = SearchQuery(**messages[-1]["content"])

    papers = search_arxiv(query.keywords, query.from_date, query.to_date)
    state.papers = papers
    state.next_step = "summarize"

    # Save state
    memory_saver.save(state.session_id, state.dict())
    return state

def search_papers_interface(keywords: str, intent: str, from_date: str,
                            to_date: str = datetime.now().strftime("%Y-%m-%d"),
                            k: int = 10):
    papers = search_arxiv(keywords, from_date, to_date, k)



# Create Gradio interface
iface = gr.Interface(
    fn=search_papers_interface,
    inputs=[
        gr.Textbox(label="Search Keywords"),
        gr.Textbox(label="Search Intent"),
        gr.Textbox(label="From Date (YYYY-MM-DD)"),
        gr.Textbox(label="To Date (YYYY-MM-DD)", value=datetime.now().strftime("%Y-%m-%d")),
        gr.Number(label="Number of Results", value=10, minimum=1, maximum=20)
    ],
    outputs=gr.Textbox(label="Results"),
    title="Research Paper Search Agent",
    description="Search and analyze arXiv papers based on keywords and intent using local LLM"
)


if __name__ == "__main__":
    iface.launch()
