import json
import http.server
import socketserver
import threading
import webbrowser
from urllib.parse import parse_qs, urlparse
from typing import Sequence
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END

import model
import tools

# Initialize MemorySaver
memory_saver = MemorySaver()

def should_continue(state: model.AgentState) -> Sequence[str]:
    """
    Determines next processing step based on state.

    Args:
        state (model.AgentState): Current agent state

    Returns:
        Sequence[str]: Next step(s) to execute
    """
    if state.next_step == "stop":
        return ["end"]
    return ["explain"]

# Graph functions
def search_papers(state: model.AgentState) -> model.AgentState:
    """
    Executes paper search and updates state.

    Args:
        state (model.AgentState): Current agent state

    Returns:
        model.AgentState: Updated state with search results
    """
    messages = state.messages
    query = model.SearchQuery(**messages[-1]["content"])

    # Create search query string
    search_query = f"site:arxiv.org {query.keywords} {query.from_date}..{query.to_date} "

    # Create input dictionary and convert to JSON
    search_inputs = {
        'search_query': search_query,
        'max_results': query.max_results
    }

    # Call the tool with JSON string containing both arguments
    papers = tools.search_arxiv(json.dumps(search_inputs))
    state.papers = papers
    state.next_step = "summarize"

    return state

def create_summaries(state: model.AgentState) -> model.AgentState:
    """
    Generates paper summaries and updates state.

    Args:
        state (model.AgentState): Current agent state

    Returns:
        model.AgentState: Updated state with paper summaries
    """
    # Convert papers to JSON string
    papers_json = json.dumps([{
        'title': p.title,
        'link': p.link,
        'snippet': p.snippet
    } for p in state.papers])

    state.summaries = tools.summarize_papers(papers_json)
    state.next_step = "explain"
    return state

def create_explanations(state: model.AgentState) -> model.AgentState:
    """
    Generates relevance explanations and updates state.

    Args:
        state (model.AgentState): Current agent state

    Returns:
        model.AgentState: Updated state with explanations
    """
    query = model.SearchQuery(**state.messages[-1]["content"])

    # Convert data to JSON string
    data = {
        'summaries': [{
            'title': s.title,
            'link': s.link,
            'summary': s.summary
        } for s in state.summaries],
        'intent': query.intent
    }

    state.explanations = tools.explain_relevance(json.dumps(data))
    state.next_step = "rank"
    return state

def rank_papers(state: model.AgentState) -> model.AgentState:
    """
    Ranks papers by relevance and updates state.

    Args:
        state (model.AgentState): Current agent state

    Returns:
        model.AgentState: Updated state with paper rankings
    """
    query = model.SearchQuery(**state.messages[-1]["content"])

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

    rankings = tools.rank_results(json.dumps(data))
    state.rankings = rankings
    state.next_step = "end"
    return state

def human_feedback(state: model.AgentState) -> model.AgentState:
    """
    Collects human feedback via web interface.
    
    Args:
        state (model.AgentState): Current agent state
        
    Returns:
        model.AgentState: Updated state based on human feedback
    """

    query = model.SearchQuery(**state.messages[-1]["content"])
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
    """
    Creates the workflow graph for paper search process.

    Returns:
        StateGraph: Compiled workflow graph with checkpointing
    """
    workflow = StateGraph(model.AgentState)

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
