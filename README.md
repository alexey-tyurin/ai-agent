# AI Research Paper Search Agent

An intelligent agent that searches, analyzes, and ranks arXiv research papers using LangGraph, LangChain, OpenAI GPT-4, and DuckDuckGo. 

The assistant helps researchers find relevant papers by understanding search intent and providing iterative refinement of search results.

## Features

- **Intelligent Paper Search**: Searches arXiv papers using DuckDuckGo with date range filtering
- **Automated Analysis**: Generates paper summaries and explains relevance to search intent
- **Smart Ranking**: Ranks papers based on relevance to research goals using LLM
- **Interactive Refinement**: Allows users to refine search intent based on initial results
- **Web Interface**: Clean Gradio interface for easy interaction
- **Iterative Search**: Supports up to 3 iterations of intent refinement per search session

## Technical Stack

- **LangGraph**: For building the AI agent workflow
- **OpenAI GPT-4**: For paper analysis and ranking
- **DuckDuckGo Search**: For paper discovery
- **Gradio**: For web interface
- **Beautiful Soup**: For paper abstract extraction
- **Pydantic**: For data validation and modeling

## Installation

1. Clone the repository:
```bash
git clone https://github.com/alexey-tyurin/ai-agent.git
cd ai-agent
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
export OPENAI_API_KEY="your-api-key"
```

Optional for debugging:
```bash
export LANGCHAIN_API_KEY="your-langchain-key"
export LANGCHAIN_TRACING_V2="true"
export LANGCHAIN_PROJECT="ai-agent"
```

## Usage

1. Start the application:
```bash
python main.py
```

2. Access the web interface at `http://localhost:7860`

3. Enter search parameters:
    - Keywords: Search terms for papers
    - Intent: Research goal or specific interest
    - Date range: Filter papers by publication date
    - Number of results: How many papers to analyze (1-20)

4. Review results and optionally:
    - Continue search with refined intent
    - Stop search when satisfied with results

## Architecture

### Components

1. **Search Pipeline**:
    - Paper search using DuckDuckGo
    - Abstract extraction from arXiv
    - Paper summarization
    - Relevance analysis
    - Result ranking

2. **State Management**:
    - Session tracking
    - State persistence
    - Iterative refinement

3. **Models**:
    - `Paper`: Basic paper metadata
    - `PaperSummary`: Includes AI-generated summary
    - `PaperExplanation`: Adds relevance explanation
    - `RankedPaper`: Includes relevance score
    - `SearchQuery`: Search parameters and session info
    - `AgentState`: Workflow state tracking

### Workflow

1. **Search**: Find papers matching keywords and date range
2. **Summarize**: Generate concise summaries
3. **Explain**: Analyze relevance to search intent
4. **Rank**: Score papers based on relevance
5. **Review**: Present results for user feedback
6. **Refine**: Optionally continue with new intent

## Error Handling

- Rate limiting protection for DuckDuckGo searches
- Validation for date formats and ranges
- Error handling for paper abstract extraction
- Score normalization for paper rankings

## Development Notes

- Maximum 3 iterations per search session
- Scores normalized to 0-10 range
- Configurable result limit (1-20 papers)
- Asynchronous processing for LLM operations

## Future Improvements

- Add support for additional paper repositories
- Implement citation analysis
- Add paper clustering by topic
- Enhance ranking algorithm
- Add export functionality for results
- Implement user authentication
- Add result caching

## Screenshots of the flow

### 1) Initial screen
<p>
        <img alt="ai-agent" src="https://github.com/alexey-tyurin/ai-agent/blob/main/screenshots/screen1.png?raw=true" width="auto" height="auto"></img> 
</p>



### 2) First results
<p>
        <img alt="ai-agent" src="https://github.com/alexey-tyurin/ai-agent/blob/main/screenshots/screen2.png?raw=true" width="auto" height="auto"></img> 
</p>


### 3) Changing intent
<p>
        <img alt="ai-agent" src="https://github.com/alexey-tyurin/ai-agent/blob/main/screenshots/screen3.png?raw=true" width="auto" height="auto"></img> 
</p>


### 4) Continue search with new intent
<p>
        <img alt="ai-agent" src="https://github.com/alexey-tyurin/ai-agent/blob/main/screenshots/screen4.png?raw=true" width="auto" height="auto"></img> 
</p>


### 5) Final results with updated intent
<p>
        <img alt="ai-agent" src="https://github.com/alexey-tyurin/ai-agent/blob/main/screenshots/screen5.png?raw=true" width="auto" height="auto"></img> 
</p>

## Acknowledgments

### References
https://www.deeplearning.ai/the-batch/how-agents-can-improve-llm-performance/

https://www.deeplearning.ai/the-batch/agentic-design-patterns-part-2-reflection

https://www.deeplearning.ai/the-batch/agentic-design-patterns-part-3-tool-use/

https://www.deeplearning.ai/the-batch/agentic-design-patterns-part-4-planning/

https://www.deeplearning.ai/the-batch/agentic-design-patterns-part-5-multi-agent-collaboration/

https://academy.langchain.com/courses/intro-to-langgraph

https://langchain-ai.github.io/langgraph/

https://smith.langchain.com/

### My certificate for LangGraph
https://academy.langchain.com/certificates/oylo2acrsw

## Contact Information

For any questions or feedback, please contact Alexey Tyurin at altyurin3@gmail.com.