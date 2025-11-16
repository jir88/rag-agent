import json

from typing import TypedDict, List, Dict, Any, Optional, Tuple
from pydantic import BaseModel
from litellm import completion
from langgraph.graph import StateGraph, START, END

from tools import Tool

import phoenix as px
from phoenix.otel import register

from openinference.instrumentation.litellm import LiteLLMInstrumentor
from opentelemetry import trace

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

# set up Arize Phoenix tracing
# session = px.launch_app()
endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = register(
    project_name="researcher-agent",  # Default is 'default'
    auto_instrument=True,  # Auto-instrument your app based on installed OI dependencies
    endpoint=endpoint,
    batch=True,
)
# now that we've set up a provider, grab the actual tracer being used
tracer = trace.get_tracer(__name__)

# ------------------------- Agent State -----------------------


class ResearchState(TypedDict):
    # the LiteLLM path to the inference client we're using to talk to an LLM
    llm: str
    # optional API key for inference client
    api_key: Optional[str] = None
    # optional URL where inference client is located
    base_url: Optional[str] = None
    # dict of tools the agent can use, mapping names to tool objects
    tools: Dict[str, Tool] = {}
    # the user's question that we're trying to answer
    user_query: Optional[str]
    # number of search rounds we've tried
    n_search_rounds: int = 0
    # maximum number of search rounds
    max_search_rounds: int = 10
    # the number of search results to return from a search
    num_pubmed_results: int = 10
    # whether we are still searching
    mode: Optional[str]
    # PubMed queries the agent has tried
    pubmed_queries: Optional[List[str]]
    # the number of 'pages' searched for each query
    pubmed_query_pages: Optional[List[int]]
    # current PubMed query
    current_pubmed_query: Optional[str]
    # list of new PMIDs returned by the current query
    current_result_ids: Optional[List[str]]
    # dict of dicts containing article metadata, keys are PMIDs
    articles: Optional[Dict[str, Dict[str, Any]]]
    # dict of article summaries, keys are PMIDs
    article_summaries: Optional[Dict[str, str]]
    # researcher notes at each step
    researcher_notes: Optional[List[str]]


def start_research(state: ResearchState):
    """
    Introduce the AI agent to the user's query and prompt it to begin research.
    """
    # format prompts as though agent is a grad student reporting back to PI/user
    # PI/user formats prompts as requests for research
    main_research_prompt = (
        "You are an intelligent, skeptical graduate student at a prestigious university. \
    You excel at critical thinking. Your thesis advisor has asked you to do a literature search and \
    summarize the results in a detailed report."
    )

    # initial prompt to start the research process
    initial_query_prompt = "Here is your thesis advisor's request:\n \
    {research_topic}\n\n \
    Before starting, please formulate a concise, high-level outline of the topics you will need to cover \
    in your literature review. Outline only the topics themselves. No extra commentary."

    # canned researcher 'response' before formulating the PubMed query
    initial_researcher_response = (
        "I will begin by searching PubMed for articles on this topic, starting \
    with general reviews of the field."
    )

    # get the LLM
    llm = state["llm"]
    # prompt for the first step
    # we need to add the main prompt to the LLM
    with guidance.system():
        llm += main_research_prompt
    # now add the research instruction
    with guidance.user():
        llm += initial_query_prompt.format(research_topic=state["user_query"])
    # add a canned researcher 'response'
    with guidance.assistant():
        llm += guidance.gen(name="plan", max_tokens=1024, stop=["<end_of_turn>"])
        # llm += initial_researcher_response
    researcher_notes = llm["plan"]
    return {
        "llm": llm,
        "n_search_rounds": state["n_search_rounds"] + 1,
        "researcher_notes": [researcher_notes],
        "mode": "search",
    }
