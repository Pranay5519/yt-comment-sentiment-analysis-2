from langgraph.graph import StateGraph, START, END
from .schemas import TopicState
from .topic_nodes import discover_topics, classify_comments

graph = StateGraph(TopicState)

graph.add_node("discover_topics", discover_topics)
graph.add_node("classify_comments", classify_comments)

graph.add_edge(START, "discover_topics")
graph.add_edge("discover_topics", "classify_comments")
graph.add_edge("classify_comments", END)

topic_graph = graph.compile()