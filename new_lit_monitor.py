import json

from typing import TypedDict, List, Dict, Any, Optional, Annotated
from pydantic import BaseModel, Field
from litellm import completion
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
import operator

from tools import PubmedSearchTool

from phoenix.otel import register
from opentelemetry import trace
from opentelemetry import trace as trace_api  # SDK for creating traces

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

class LitMonitorState(TypedDict):
    llm: str
    """The LiteLLM path to the inference client we're using to talk to an LLM."""
    # optional API key for inference client
    api_key: Optional[str] = None
    # optional URL where inference client is located
    base_url: Optional[str] = None
    topic_description: str
    """A description of the topic the user is interested in."""
    prior_pmids: List[str] = []
    """A list of the PMIDs we have seen before."""
    article_evaluations: Annotated[List[Dict[str, Any]], operator.add]
    """A list of dicts containing the metadata about each paper and our evaluation of it."""
    n_search_rounds: int = 0
    """The number of search result pages we've looked at thus far."""
    max_search_rounds: int = 3
    """The maximum number of pages we can look at before stopping."""
    num_pubmed_results: int = 10
    """The maximum number of search results per page."""

def do_search(state: LitMonitorState):
    """
    Do the initial PubMed search.
    """
    result_list = [1, 2, 3]
    return [Send(node="eval_paper", arg={'pmid': id}) for id in result_list]

def eval_paper(state):
    """
    Evaluate the relevance of a single new paper.
    """
    pmid = state['pmid']
    article_evaluation = {
        'pmid': pmid,
        'evaluation': "Paper " + str(pmid) + " is irrelevant! >:(",
    }
    # return evaluation wrapped in list so it can be appended correctly
    return {
        "article_evaluations": [article_evaluation],
    }

class LitMonitor:
    """
    Workflow powered by LiteLLM that gets new papers from PubMed and decides
    whether they are relevant to the user's interests.
    """

    def __init__(self, llm:str, api_key:str = None, base_url:str = None):
        """
        Create a new monitor workflow with a back-end LLM.

        Args:
            llm (str): LiteLLM identifier of the model to use for LLM inference
            api_key (str): Optional API key if the LLM service requires it
            base_url (str): Optional custom URL where LLM service is located
        """
        self.llm = llm
        self.api_key = api_key
        self.base_url = base_url

        # build the agent graph
        self._initialize_agent_graph()
    
    def check_search(self, query: str):
        # build agent state
        state = {
            'llm': self.llm,
            'api_key': self.api_key,
            'base_url': self.base_url,
            'topic_description': query,
            'prior_pmids': [],
            'article_evaluations': [],
            'n_search_rounds': 0,
            'max_search_rounds': 3,
            'num_pubmed_results': 10
        }
        with tracer.start_as_current_span("Invoke ResearchAgent") as session_span:
            final_result = self.agent_graph.invoke(
                input=state,
                config={ "recursion_limit": 100 }
            )
            session_span.set_status(trace_api.Status(trace_api.StatusCode.OK))
        print("Agent finished!")
        return final_result

    def _initialize_agent_graph(self):
        """
        Assemble the agent graph and compile it.
        """
        # construct graph object
        self.agent_graph = StateGraph(LitMonitorState)

        # define nodes
        self.agent_graph.add_node("eval_paper", eval_paper)
        
        # define edges
        self.agent_graph.add_conditional_edges(
            source=START,
            path=do_search
        )
        self.agent_graph.add_edge("eval_paper", END)
        
        # compile graph
        self.agent_graph = self.agent_graph.compile()

if __name__ == "__main__":
    # create the agent itself
    # model="huggingface/Qwen/Qwen2.5-Coder-32B-Instruct"
    # api_key=None
    # base_url=None
    # model="openai/gemma-3n-E4B-it-UD-Q5_K_XL-cpu"
    model="openai/granite-4.0-h-tiny-UD-Q5_K_XL-cpu"
    # model="openai/gemma-3-4B-it-UD-Q4_K_XL-cpu"
    api_key="sk-fake"
    base_url="http://127.0.0.1:8080/v1"

    agent = LitMonitor(
        llm=model,
        api_key=api_key,
        base_url=base_url
    )

    query = (
        "Summarize recent research on bile acid metabolism, focusing on particular areas of active research. "
        "Begin with a general introduction to bile acid metabolism. "
        "Look for both host and microbiome interactions with bile acids and the effects of these interactions on disease."
    )
    result = agent.check_search(query=query)
    print(json.dumps(result, indent=2))