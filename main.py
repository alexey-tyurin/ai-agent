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

def search_papers_interface(keywords: str, intent: str, from_date: str,
                            to_date: str = datetime.now().strftime("%Y-%m-%d"),
                            k: int = 10):
    print(f"keywords={keywords}, intent={intent}, from_date={from_date}, to_date={to_date}, k={k}")


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