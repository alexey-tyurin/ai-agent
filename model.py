
import uuid
from datetime import datetime
from typing import List, Dict, TypedDict
from pydantic import BaseModel, Field, validator

__all__ = ["Paper", "PaperSummary", "PaperExplanation", "RankedPaper", "AgentState", "SearchQuery", "HumanDecision"]

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
    max_results: int = Field(default=10, ge=1, le=20)
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

class HumanDecision(TypedDict):
    decision: str
