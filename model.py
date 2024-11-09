
import uuid
from datetime import datetime
from typing import List, Dict, TypedDict
from pydantic import BaseModel, Field, validator

__all__ = ["Paper", "PaperSummary", "PaperExplanation", "RankedPaper", "AgentState", "SearchQuery", "HumanDecision"]

# Pydantic Models
class Paper(BaseModel):
    """
    Represents a basic paper with title, link and snippet.

    Attributes:
        title (str): Paper title
        link (str): URL to the paper
        snippet (str): Brief excerpt or abstract
    """
    title: str
    link: str
    snippet: str

class PaperSummary(BaseModel):
    """
    Represents a summarized paper.

    Attributes:
        title (str): Paper title
        link (str): URL to the paper
        summary (str): AI-generated summary of the paper
    """
    title: str
    link: str
    summary: str

class PaperExplanation(BaseModel):
    """
    Represents a paper with relevance explanation.

    Attributes:
        title (str): Paper title
        link (str): URL to the paper
        summary (str): Paper summary
        explanation (str): Explanation of relevance to search intent
    """
    title: str
    link: str
    summary: str
    explanation: str

class RankedPaper(BaseModel):
    """
    Represents a paper with relevance ranking.

    Attributes:
        title (str): Paper title
        link (str): URL to the paper
        summary (str): Paper summary
        explanation (str): Relevance explanation
        score (float): Relevance score (0-10)
    """
    title: str
    link: str
    summary: str
    explanation: str
    score: float = Field(ge=0, le=10)

class SearchQuery(BaseModel):
    """
    Represents a search query with validation.

    Attributes:
        keywords (str): Search terms
        intent (str): Search intent/goal
        from_date (str): Start date (YYYY-MM-DD)
        to_date (str): End date (YYYY-MM-DD)
        max_results (int): Maximum results to return (1-20)
        session_id (str): Unique session identifier
    """
    keywords: str
    intent: str
    from_date: str
    to_date: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    max_results: int = Field(default=10, ge=1, le=20)
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    @validator('from_date', 'to_date')
    def validate_date(cls, v):
        """
        Validates date format.

        Args:
            v (str): Date string to validate

        Returns:
            str: Validated date string

        Raises:
            ValueError: If date format is invalid
        """
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Date must be in YYYY-MM-DD format")
        return v

class AgentState(BaseModel):
    """
    Represents the current state of the search agent.

    Attributes:
        messages (List[Dict]): Conversation history
        papers (List[Paper]): Found papers
        summaries (List[PaperSummary]): Paper summaries
        explanations (List[PaperExplanation]): Relevance explanations
        rankings (List[RankedPaper]): Ranked papers
        next_step (str): Next processing step
        session_id (str): Session identifier
    """
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
        """
        Custom dictionary conversion handling Pydantic models in lists.

        Returns:
            dict: State as dictionary with all nested models converted
        """
        d = super().dict(*args, **kwargs)
        d['papers'] = [paper.dict() for paper in self.papers]
        d['summaries'] = [summary.dict() for summary in self.summaries]
        d['explanations'] = [exp.dict() for exp in self.explanations]
        d['rankings'] = [rank.dict() for rank in self.rankings]
        return d

class HumanDecision(TypedDict):
    decision: str
