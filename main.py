from datetime import datetime
import gradio as gr
import uuid
import model
import graph

# Set API keys for LangSmith
# Uncomment these lines if debugging / profiling is necessary
# _set_env("LANGCHAIN_API_KEY")
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_PROJECT"] = "ai-agent"


def search_papers_interface(keywords: str, intent: str, from_date: str,
                            to_date: str = datetime.now().strftime("%Y-%m-%d"),
                            max_results: int = 10):
    """
    Creates a search interface for arXiv papers based on user inputs.

    Args:
        keywords (str): Search terms to find relevant papers
        intent (str): User's research intent/goal
        from_date (str): Start date for paper search (YYYY-MM-DD)
        to_date (str): End date for paper search (YYYY-MM-DD)
        max_results (int): Maximum number of papers to return (default: 10)

    Returns:
        dict: Contains results_box with formatted search results and intent_box with final search intent

    Raises:
        Exception: Any errors during the search process
    """

    session_id = str(uuid.uuid4())

    # Validate inputs using Pydantic
    query = model.SearchQuery(
        keywords=keywords,
        intent=intent,
        from_date=from_date,
        to_date=to_date,
        max_results=max_results,
        session_id=session_id
    )

    gr = graph.create_graph()

    # Initialize state
    state = model.AgentState(
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
            current_state = gr.invoke(current_state, {"configurable": {"thread_id": "1"}})

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