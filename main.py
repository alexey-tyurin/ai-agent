from typing import List, Dict, Optional
from datetime import datetime
from duckduckgo_search import DDGS
from duckduckgo_search.exceptions import RatelimitException
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
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
import getpass
import os
import asyncio
import time
from typing import Annotated, Sequence, TypedDict, Union
from random import uniform
import http.server
import socketserver
import threading
import webbrowser
from urllib.parse import parse_qs, urlparse

from model import Paper, PaperSummary, PaperExplanation, RankedPaper, AgentState, SearchQuery, HumanDecision

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

# Set API keys for OpenAI
_set_env("OPENAI_API_KEY")

# Set API keys for LangSmith
# Uncomment these lines if debugging / profiling is necessary
# _set_env("LANGCHAIN_API_KEY")
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_PROJECT"] = "ai-agent"

# Initialize ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Initialize MemorySaver
memory_saver = MemorySaver()


def should_continue(state: AgentState) -> Sequence[str]:
    """Determine the next node based on human decision"""
    if state.next_step == "stop":
        return ["end"]
    return ["explain"]

def normalize_arxiv_url(url):
    # Replace both pdf and html versions with abstract version
    normalized_url = re.sub(r'arxiv\.org/(pdf|html)/', 'arxiv.org/abs/', url)
    return normalized_url

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

# Tools
@tool
def search_arxiv(inputs: str) -> List[Paper]:
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
                papers.append(Paper(
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
def summarize_papers(text: str) -> List[PaperSummary]:
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
            PaperSummary(
                title=paper['title'],
                link=paper['link'],
                summary=summary
            )
            for paper, summary in zip(papers, summaries_content)
        ]

    return run_async(process_papers())

@tool
def explain_relevance(data: str) -> List[PaperExplanation]:
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
            PaperExplanation(
                title=summary['title'],
                link=summary['link'],
                summary=summary['summary'],
                explanation=explanation
            )
            for summary, explanation in zip(summaries, explanations_content)
        ]

    return run_async(process_explanations())

@tool
def rank_results(data: str) -> List[RankedPaper]:
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

            rankings.append(RankedPaper(
                title=exp['title'],
                link=exp['link'],
                summary=exp['summary'],
                explanation=exp['explanation'],
                score=score
            ))

        return sorted(rankings, key=lambda x: x.score, reverse=True)

    return run_async(process_rankings())

# Graph functions
def search_papers(state: AgentState) -> AgentState:
    messages = state.messages
    query = SearchQuery(**messages[-1]["content"])

    # Create search query string
    search_query = f"site:arxiv.org {query.keywords} {query.from_date}..{query.to_date} "

    # Create input dictionary and convert to JSON
    search_inputs = {
        'search_query': search_query,
        'max_results': query.max_results  # You can adjust this or make it part of SearchQuery
    }

    # Call the tool with JSON string containing both arguments
    papers = search_arxiv(json.dumps(search_inputs))
    state.papers = papers
    state.next_step = "summarize"

    return state

def create_summaries(state: AgentState) -> AgentState:
    # Convert papers to JSON string
    papers_json = json.dumps([{
        'title': p.title,
        'link': p.link,
        'snippet': p.snippet
    } for p in state.papers])

    state.summaries = summarize_papers(papers_json)
    state.next_step = "explain"
    return state

def create_explanations(state: AgentState) -> AgentState:
    query = SearchQuery(**state.messages[-1]["content"])

    # Convert data to JSON string
    data = {
        'summaries': [{
            'title': s.title,
            'link': s.link,
            'summary': s.summary
        } for s in state.summaries],
        'intent': query.intent
    }

    state.explanations = explain_relevance(json.dumps(data))
    state.next_step = "rank"
    return state

def rank_papers(state: AgentState) -> AgentState:
    query = SearchQuery(**state.messages[-1]["content"])

    # Convert data to JSON string
    data = {
        'explanations': [{
            'title': e.title,
            'link': e.link,
            'summary': e.summary,
            'explanation': e.explanation
        } for e in state.explanations],
        'intent': query.intent
    }

    rankings = rank_results(json.dumps(data))
    state.rankings = rankings
    state.next_step = "end"
    return state

def human_feedback(state: AgentState) -> AgentState:
    """Get human feedback after reviewing ranked results using a simple HTTP server"""
    query = SearchQuery(**state.messages[-1]["content"])
    feedback = {"choice": None, "new_intent": None}

    # Define CSS separately with doubled curly braces to escape them
    css = """
        body {{ font-family: Arial, sans-serif; margin: 20px; max-width: 800px; margin: 0 auto; padding: 20px; }}
        .paper {{ border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .button-container {{ margin-top: 20px; text-align: center; }}
        button {{ padding: 10px 20px; margin: 0 10px; font-size: 16px; cursor: pointer; }}
        .intent-form {{ display: none; margin-top: 20px; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }}
        .intent-input {{ width: 100%; padding: 10px; margin: 10px 0; }}
    """

    # Create HTML content
    html_content = f"""
    <html>
    <head>
        <title>Review Results</title>
        <style>
            {css}
        </style>
        <script>
            function showIntentForm() {{
                document.getElementById('intentForm').style.display = 'block';
                document.getElementById('initialButtons').style.display = 'none';
            }}
            
            function submitIntent() {{
                const newIntent = document.getElementById('newIntent').value.trim();
                if (newIntent) {{
                    window.location.href = '/decision?choice=continue&new_intent=' + encodeURIComponent(newIntent);
                }} else {{
                    alert('Please enter a new intent');
                }}
            }}
        </script>
    </head>
    <body>
        <h2>Current Intent: {query.intent}</h2>
        <h3>Ranked Results:</h3>
        
        {"".join(f'''
        <div class="paper">
            <h4>{rank.title}</h4>
            <p><strong>Link:</strong> <a href="{rank.link}" target="_blank">{rank.link}</a></p>
            <p><strong>Score:</strong> {rank.score}/10</p>
            <p><strong>Summary:</strong><br>{rank.summary}</p>
            <p><strong>Relevance to Intent:</strong><br>{rank.explanation}</p>
        </div>
        ''' for rank in state.rankings)}
        
        <div id="initialButtons" class="button-container">
            <button onclick="showIntentForm()">Continue with New Intent</button>
            <button onclick="window.location.href='/decision?choice=stop'">Stop Search</button>
        </div>
        
        <div id="intentForm" class="intent-form">
            <h3>Enter New Intent</h3>
            <input type="text" id="newIntent" class="intent-input" 
                   placeholder="Enter new search intent..."
                   value="{query.intent}">
            <div class="button-container">
                <button onclick="submitIntent()">Continue Search</button>
                <button onclick="window.location.href='/decision?choice=stop'">Cancel</button>
            </div>
        </div>
    </body>
    </html>
    """

    class FeedbackHandler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith('/decision'):
                query_components = parse_qs(urlparse(self.path).query)
                choice = query_components.get('choice', [''])[0]
                if choice in ['continue', 'stop']:
                    feedback['choice'] = choice
                    if choice == 'continue':
                        feedback['new_intent'] = query_components.get('new_intent', [query.intent])[0]

                    self.send_response(200)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()
                    self.wfile.write(b"Decision recorded. You can close this window.")
                    # Stop the server
                    threading.Thread(target=lambda: self.server.shutdown()).start()
            else:
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(html_content.encode())

    # Find an available port
    def get_free_port():
        with socketserver.TCPServer(("", 0), None) as s:
            return s.server_address[1]

    port = get_free_port()

    # Start server
    with socketserver.TCPServer(("", port), FeedbackHandler) as httpd:
        # Open browser
        webbrowser.open(f'http://localhost:{port}')

        # Serve until decision is made
        while feedback['choice'] is None:
            httpd.handle_request()

    # Update state based on decision
    if feedback['choice'] == 'stop':
        state.next_step = "stop"
    else:
        state.next_step = "explain"
        # Update the intent in the messages
        messages = state.messages
        content = messages[-1]["content"]
        content["intent"] = feedback['new_intent']
        messages[-1]["content"] = content
        state.messages = messages

    return state

# Create the graph
def create_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("search", search_papers)
    workflow.add_node("summarize", create_summaries)
    workflow.add_node("explain", create_explanations)
    workflow.add_node("rank", rank_papers)
    workflow.add_node("human_feedback", human_feedback)

    workflow.set_entry_point("search")

    workflow.add_edge("search", "summarize")
    workflow.add_edge("summarize", "explain")
    workflow.add_edge("explain", "rank")
    workflow.add_edge("rank", "human_feedback")

    # Add conditional edge from human_feedback
    workflow.add_conditional_edges(
        "human_feedback",
        should_continue,
        {
            "explain": "explain",  # Loop back to explain step if continuing
            "end": END,
        }
    )

    # Compile graph with persistence for checkpoints
    return workflow.compile(checkpointer=memory_saver)


def search_papers_interface(keywords: str, intent: str, from_date: str,
                            to_date: str = datetime.now().strftime("%Y-%m-%d"),
                            max_results: int = 10):

    session_id = str(uuid.uuid4())

    # Validate inputs using Pydantic
    query = SearchQuery(
        keywords=keywords,
        intent=intent,
        from_date=from_date,
        to_date=to_date,
        max_results=max_results,
        session_id=session_id
    )

    graph = create_graph()

    # Initialize state
    state = AgentState(
        messages=[{"content": query.dict()}],
        next_step="search",
        session_id=session_id
    )

    try:
        # Initialize results storage
        all_results = []
        iteration = 1
        current_state = state

        # Run iterations
        while iteration <= 3:  # Maximum 3 iterations
            # Run the graph
            current_state = graph.invoke(current_state, {"configurable": {"thread_id": "1"}})

            # Get current intent from the state dictionary
            state_dict = dict(current_state)
            current_messages = state_dict.get('messages', [])
            if current_messages:
                current_content = current_messages[-1].get('content', {})
                current_intent = current_content.get('intent', intent)  # Fallback to original intent
            else:
                current_intent = intent

            # Format results for this iteration
            results = [f"\nIteration {iteration} Results (Intent: {current_intent}):"]
            for rank in current_state["rankings"]:
                results.append(f"""
Title: {rank.title}
Link: {rank.link}
Score: {rank.score}/10

Summary:
{rank.summary}

Relevance to Intent:
{rank.explanation}
---------------------""")

            all_results.extend(results)

            # Check if we should continue
            if current_state["next_step"] == "stop":
                all_results.append("\nSearch process stopped by user.")
                break

            iteration += 1
            if iteration > 3:
                all_results.append("\nReached maximum number of iterations (3).")

        # Get final intent for updating the interface
        final_state_dict = dict(current_state)
        final_messages = final_state_dict.get('messages', [])
        if final_messages:
            final_content = final_messages[-1].get('content', {})
            final_intent = final_content.get('intent', intent)
        else:
            final_intent = intent

        # Return both results and the updated intent
        return {
            results_box: "\n".join(all_results),
            intent_box: final_intent
        }

    except Exception as e:
        raise e

# Create Gradio interface with proper component references
with gr.Blocks(title="Research Paper Search Agent") as iface:
    gr.Markdown("# Research Paper Search Agent")
    gr.Markdown("Search and analyze arXiv papers based on keywords and intent using LLM")

    with gr.Row():
        with gr.Column():
            keywords_box = gr.Textbox(label="Search Keywords")
            intent_box = gr.Textbox(label="Search Intent")
            date_from_box = gr.Textbox(label="From Date (YYYY-MM-DD)")
            date_to_box = gr.Textbox(
                label="To Date (YYYY-MM-DD)",
                value=datetime.now().strftime("%Y-%m-%d")
            )
            results_count = gr.Number(
                label="Number of Results (1 - 20)",
                value=2,
                minimum=1,
                maximum=20
            )

            search_button = gr.Button("Search")

    results_box = gr.Textbox(label="Results")

    # Connect the interface components
    search_button.click(
        fn=search_papers_interface,
        inputs=[
            keywords_box,
            intent_box,
            date_from_box,
            date_to_box,
            results_count
        ],
        outputs=[
            results_box,
            intent_box
        ]
    )

if __name__ == "__main__":
    iface.launch()