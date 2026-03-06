from src.comments_classification.schemas import (
    TopicClassificationOutput,
    TopicState,
    ClassifiedComment,
    TopicDiscoveryOutput
)
from langgraph.graph import StateGraph, START, END
from .schemas import TopicState
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()


class TopicClassifier:

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash", temperature: float = 0.0):
        """
        Initialize the LLM and structured output parsers
        """
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature

        # LLM with API key
        self.llm = ChatGoogleGenerativeAI(model=self.model_name,
                                          temperature=self.temperature,
                                          api_key=self.api_key)

        # structured outputs
        self.topic_discovery_llm = self.llm.with_structured_output(TopicDiscoveryOutput)
        self.topic_classification_llm = self.llm.with_structured_output(TopicClassificationOutput)

    def _discover_topics(self, state: TopicState):
        """
        Discover discussion topics from comments
        """

        comments_text = "\n".join(state["comments"][:100])

        prompt = f"""
You are analyzing YouTube comments.

TASK:
- Create discussion topics from the comments
- Topics must be created by you
- 2–3 words per topic
- No sentiment words
- Max 8 topics
- Dont Include "Other" as a topic

COMMENTS:
{comments_text}
"""

        response = self.topic_discovery_llm.invoke(prompt)

        return {
            **state,
            "topics": response.topics
        }

    def _classify_comments(self, state: TopicState):
        """
        Classify comments into discovered topics
        """

        comments_text = "\n".join(state["comments"])
        topics_text = ", ".join(state["topics"])

        prompt = f"""
You are classifying comments into topics.

TOPICS:
{topics_text}

RULES:
- Use ONLY the provided topics
- One topic per comment
- Do not invent new topics
- No explanations

COMMENTS:
{comments_text}
"""

        response = self.topic_classification_llm.invoke(prompt)

        return {
            **state,
            "classified_comments": [
                item.model_dump() for item in response.results
            ]
        }
            
    def graph(self):
        graph = StateGraph(TopicState)

        graph.add_node("discover_topics", self._discover_topics)
        graph.add_node("classify_comments", self._classify_comments)

        graph.add_edge(START, "discover_topics")
        graph.add_edge("discover_topics", "classify_comments")
        graph.add_edge("classify_comments", END)

        topic_graph = graph.compile()

        return topic_graph