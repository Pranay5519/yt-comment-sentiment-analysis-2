from src.comments_classification.schemas import (
    TopicClassificationOutput,
    TopicState,
    ClassifiedComment,
    TopicDiscoveryOutput,
    TopicSummary,
    TopicSummaryOutput
)
from langgraph.graph import StateGraph, START, END
from .schemas import TopicState
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()


class TopicClassifier:

    def __init__(self, api_key: str,min_comment_count_for_summary: int = 4 ,  model_name: str = "gemini-2.5-flash", temperature: float = 0.0):
        """
        Initialize the LLM and structured output parsers
        """
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.min_comment_count_for_summary = min_comment_count_for_summary
        # LLM with API key
        self.llm = ChatGoogleGenerativeAI(model=self.model_name,
                                          temperature=self.temperature,
                                          api_key=self.api_key)

        # structured outputs
        self.topic_discovery_llm = self.llm.with_structured_output(TopicDiscoveryOutput)
        self.topic_classification_llm = self.llm.with_structured_output(TopicClassificationOutput)

    def _discover_topics(self, state: TopicState):
        """
        Discover discussion topics from comments OR use user-provided topics
        """

        # If user manually provides topics → use them
        if state["user_topics"] and len(state["user_topics"]) > 0:
            return {
                **state,
                "topics": state["user_topics"]
            }

        # Otherwise discover topics using LLM
        comments_text = "\n".join(state["comments"][:100])

        prompt = f"""
        You are analyzing YouTube comments.

        TASK:
        - Create discussion topics from the comments
        - Topics must be created by you
        - 2–3 words per topic
        - No sentiment words
        - Max 8 topics
        - Dont include "Other" as a topic

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
        You are classifying YouTube comments into discussion topics.

        TOPICS:
        {topics_text}

        TASK:
        Assign comments to the most relevant topic.

        RULES:
        - Use ONLY the provided topics.
        - Assign a topic ONLY if the comment clearly relates to one of the topics.
        - If the comment is irrelevant, unclear, spam, or does not meaningfully fit any topic, SKIP it.
        - It is NOT necessary to classify every comment.
        - Do not invent new topics.
        - No explanations.

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
    def topics_summary(self, state: TopicState):
        """
        Generate summaries for each discovered topic based on grouped comments
        """

        topic_comments = {}

        for item in state['classified_comments']:
            topic = item["topic"]
            comment = item["comment"]

            if topic not in topic_comments:
                topic_comments[topic] = []

            topic_comments[topic].append(comment)

        
        
        prompt_parts = []

        for topic, comments in topic_comments.items():
            if len(comments)>=self.min_comment_count_for_summary:
                comments_text = "\n".join(comments)

                prompt_parts.append(f"""
                TOPIC: {topic}

                COMMENTS:
                {comments_text}
                """)

        prompt = f"""
        You are analyzing YouTube comments grouped by discussion topic.

        TASK:
        Generate a short analytical summary for each topic.

        RULES:
        - Maximum 5 lines per topic
        - Focus on key insights, doubts, and common opinions
        - Do not repeat comments
        - Keep summaries concise and analytical

        TOPIC DATA:
        {''.join(prompt_parts)}
        """

        summary_llm = self.llm.with_structured_output(TopicSummaryOutput)

        response = summary_llm.invoke(prompt)

        return {
            **state,
            "summary": [
                item.model_dump() for item in response.final_summary
            ]
        }
        
    def summary_router(self, state: TopicState):
    
        if state.get("generate_topic_summary", False):
            return "topics_summary"
        
        return END
    def graph(self):

        graph = StateGraph(TopicState)

        graph.add_node("discover_topics", self._discover_topics)
        graph.add_node("classify_comments", self._classify_comments)
        graph.add_node("topics_summary", self.topics_summary)

        graph.add_edge(START, "discover_topics")
        graph.add_edge("discover_topics", "classify_comments")

        graph.add_conditional_edges(
            "classify_comments",
            self.summary_router
        )

        graph.add_edge("topics_summary", END)

        topic_graph = graph.compile()

        return topic_graph