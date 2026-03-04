from src.comments_classification.schemas import TopicClassificationOutput  , TopicState , ClassifiedComment ,TopicDiscoveryOutput
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv  import load_dotenv
load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.0
)

topic_discovery_llm = llm.with_structured_output(TopicDiscoveryOutput)
topic_classification_llm = llm.with_structured_output(TopicClassificationOutput)
def discover_topics(state: TopicState):
    comments_text = "\n".join(state["comments"][:100])

    prompt = f"""
You are analyzing YouTube comments.

TASK:
- Create discussion topics from the comments
- Topics must be created by you
- 2–3 words per topic
- No sentiment words
- Max 8 topics
- Dont Include  "Other" as a topic
COMMENTS:
{comments_text}
"""

    response = topic_discovery_llm.invoke(prompt)

    return {
        **state,
        "topics": response.topics
    }
def classify_comments(state: TopicState):
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

    response = topic_classification_llm.invoke(prompt)

    return {
        **state,
        "classified_comments": [
            item.model_dump() for item in response.results
        ]
    }