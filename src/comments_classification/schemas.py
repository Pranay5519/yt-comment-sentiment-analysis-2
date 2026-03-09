from pydantic import BaseModel, Field
from typing import TypedDict, List


class TopicState(TypedDict, total=False):
    """State passed between LangGraph nodes."""

    comments: List[str]
    topics: List[str]
    user_topics : List[str]
    classified_comments: List[dict]
    summary: List[dict]
    generate_topic_summary: bool


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
    
class TopicSummary(BaseModel):
    """Schema for generating a summary of comments grouped by topic."""
    
    topic: str
    summary: str = Field(
        description="Generate a concise summary (maximum 5 lines) that captures the key points, common opinions, and main discussion themes from the comments related to this topic."
    )
    
class TopicSummaryOutput(BaseModel):
    "list of Output for multiple topics"
    final_summary : List[TopicSummary]