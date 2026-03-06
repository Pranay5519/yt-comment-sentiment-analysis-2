from pydantic import BaseModel, Field
from typing import TypedDict, List


class TopicState(TypedDict):
    """State passed between LangGraph nodes containing comments, topics, and results."""
    
    comments: List[str]
    topics: List[str]
    classified_comments: List[dict]


class TopicDiscoveryOutput(BaseModel):
    """LLM output containing discovered discussion topics."""

    topics: List[str] = Field(
        description="Short, non-overlapping topic names"
    )


class ClassifiedComment(BaseModel):
    """Single classified comment with its assigned topic."""

    comment: str
    topic: str


class TopicClassificationOutput(BaseModel):
    """LLM output containing classification results for all comments."""

    results: List[ClassifiedComment]