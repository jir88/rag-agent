from typing import TypedDict, List, Dict, Any, Optional
from pydantic import BaseModel, Field
from litellm import completion
from langgraph.graph import StateGraph, START, END

from phoenix.otel import register
from opentelemetry import trace

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

class AgentPlan(BaseModel):
    """An agent's research plan as a list of steps to take."""
    
    steps: List[str] = Field(description="A list of individual steps in the agent's plan.")

    def format_readable(self) -> str:
        """Convert the plan into a well-formatted string."""
        # format steps
        step_txt = ["Step " + str(index) + ": " + step for index,step in enumerate(self.steps, start=1)]
        # join and return
        return "\n\n".join(step_txt)

# ------------------------- Agent State -----------------------


class ResearchState(TypedDict):
    # the LiteLLM path to the inference client we're using to talk to an LLM
    llm: str
    # optional API key for inference client
    api_key: Optional[str] = None
    # optional URL where inference client is located
    base_url: Optional[str] = None
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

    planning_messages = [
        {
            'role': 'system',
            'content': main_research_prompt
        },
        {
            'role': 'user',
            'content': initial_query_prompt.format(research_topic=state["user_query"])
        }
    ]
    # get the plan
    response = completion(
        model=state['llm'],
        api_key=state['api_key'],
        base_url=state['base_url'],
        messages=planning_messages,
        response_format=AgentPlan,
        stream=False,
        max_tokens=1024,
        temperature=1.0,
        top_p=0.95
    )
    
    ai_msg = response['choices'][0]['message']
    print("Raw text:" + ai_msg['content'])
    plan = AgentPlan.model_validate_json(ai_msg['content'])
    print("Initial plan:\n\n" + plan.format_readable())
    
    return {
        "n_search_rounds": state["n_search_rounds"] + 1,
        "researcher_notes": [plan.format_readable()],
        "mode": "search",
    }

class ResearchAgent:
    """
    An LLM-powered agent tuned for doing literature reviews via PubMed.
    """

    def __init__(self, llm:str, api_key:str = None, base_url:str = None):
        """
        Create a new research agent with a back-end LLM.

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
    
    def submit_question(self, question:str):
        """
        Submit a question for this agent to research.

        Args:
            question (str): The question text.
        
        Returns:
            Answer to question in text form based on a literature review via PubMed.
        """
        # build initial state
        agent_state = {
            # the inference client we're using to talk to an LLM
            "llm": self.llm,
            "api_key": self.api_key,
            "base_url": self.base_url,
            # the user's question that we're trying to answer
            "user_query": question,
            # blank plan
            "plan": [],
            # number of search rounds we've tried
            "n_search_rounds": 0,
            # maximum number of search rounds
            "max_search_rounds": 10,
            # the number of search results to return from a search
            "num_pubmed_results": 5,
            # whether we are still searching
            "mode": None,
            # PubMed queries the agent has tried
            "pubmed_queries": None,
            # the number of 'pages' searched for each query
            "pubmed_query_pages": None,
            # current PubMed query
            "current_pubmed_query": None,
            "current_result_ids": None,
            # dict of dicts containing article metadata, keys are PMIDs
            "articles": None,
            # dict of article summaries, keys are PMIDs
            "article_summaries": None,
            # researcher notes at each step
            "researcher_notes": None
        }
        with tracer.start_as_current_span("Invoke ResearchAgent"):
            final_result = self.agent_graph.invoke(agent_state)
        print("Agent finished!")
        return final_result

    def _initialize_agent_graph(self):
        """
        Assemble the agent graph and compile it.
        """
        # construct graph object
        self.agent_graph = StateGraph(ResearchState)

        # define nodes
        self.agent_graph.add_node("start_research", start_research)
        # self.agent_graph.add_node("do_research_step", do_research_step)
        # self.agent_graph.add_node("evaluate_step_success", evaluate_step_success)
        # self.agent_graph.add_node("summarize_research_result", summarize_research_result)
        # self.agent_graph.add_node("check_agent_progress", check_agent_progress)
        # self.agent_graph.add_node("generate_report", generate_report)
        
        # define edges
        self.agent_graph.add_edge(START, "start_research")
        self.agent_graph.add_edge("start_research", END)
        # self.agent_graph.add_edge("make_plan", "do_research_step")
        # self.agent_graph.add_edge("do_research_step", "summarize_research_result")
        # self.agent_graph.add_edge("summarize_research_result", "evaluate_step_success")
        # self.agent_graph.add_edge("evaluate_step_success", "check_agent_progress")
        # self.agent_graph.add_conditional_edges(
        #     "check_agent_progress",
        #     decide_next_step,
        #     {
        #         "continue": "do_research_step",
        #         "respond": "generate_report"
        #     }
        # )
        # self.agent_graph.add_edge("generate_report", END)
        
        # compile graph
        self.agent_graph = self.agent_graph.compile()