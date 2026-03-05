from pydantic import BaseModel, Field
from typing import TypedDict, List

class TopicState(TypedDict):
    comments: List[str]
    topics: List[str]
    classified_comments: List[dict]
    
class TopicDiscoveryOutput(BaseModel):
    topics: List[str] = Field(
        description="Short, non-overlapping topic names"
    )
    
class ClassifiedComment(BaseModel):
    comment: str
    topic: str


class TopicClassificationOutput(BaseModel):
    results: List[ClassifiedComment]