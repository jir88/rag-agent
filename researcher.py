from typing import TypedDict, List, Dict, Any, Optional
from pydantic import BaseModel, Field
from litellm import completion
from langgraph.graph import StateGraph, START, END

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

class TopicList(BaseModel):
    """An agent's research plan as a list of topics to research."""
    
    topics: List[str] = Field(description="A list of individual topics that need to be researched.")

    def format_readable(self) -> str:
        """Convert the list into a well-formatted string."""
        # format steps
        step_txt = ["Topic " + str(index) + ": " + step for index,step in enumerate(self.topics, start=1)]
        # join and return
        return "\n".join(step_txt)

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
    # message history for the LLM
    messages: List[Dict[str, str]] = []
    # the final report
    final_report: Optional[str] = None


def start_research(state: ResearchState):
    """
    Introduce the AI agent to the user's query and prompt it to begin research.
    """
    # format prompts as though agent is a grad student reporting back to PI/user
    # PI/user formats prompts as requests for research
    main_research_prompt = (
        "You are an intelligent, skeptical graduate student at a prestigious university. "
        "You excel at critical thinking. Your thesis advisor has asked you to do a literature search and "
        "summarize the results in a detailed report."
    )

    # initial prompt to start the research process
    initial_query_prompt = (
        "Here is your thesis advisor's request:\n"
        "{research_topic}\n\n "
        "Before starting, please formulate a concise, high-level outline of the topics you will need to cover "
        "in your literature review. Outline only the topics themselves in JSON format. No extra commentary."
    )

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
        response_format=TopicList,
        stream=False,
        max_tokens=1024,
        temperature=1.0,
        top_p=0.95
    )
    
    ai_msg = response['choices'][0]['message']
    plan = TopicList.model_validate_json(ai_msg['content'])
    formatted_plan = plan.format_readable()
    print("Initial list of topics to research:\n\n" + formatted_plan)

    # create the list of messages that will be used for downstream steps
    output_messages = [
        {
            'role': 'system',
            'content': main_research_prompt
        },
        {
            'role': 'user',
            'content': "Here is your thesis advisor's request:\n" + state["user_query"]
        },
        {
            'role': 'assistant',
            'content': "To cover all aspects of this request, I need to research the following topics:\n\n" + formatted_plan
        }
    ]
    return {
        "n_search_rounds": state["n_search_rounds"] + 1,
        "researcher_notes": [formatted_plan],
        "messages": output_messages,
        "mode": "search",
    }

def compose_pubmed_query(state: ResearchState):
    """
    Given the current state of the research, write a new PubMed search query.
    """
    # prompt the LLM to make a PubMed query
    query_compose_prompt = (
        "Given the current state of your literature review, please write down a single PubMed "
        "search query that will help fill in a gap in your literature review thus far. Prefer "
        "concise queries using only one search term. Surround the query with square brackets, "
        "[like this]. If a single-term query fails, join terms with AND, like this: "
        "[first topic AND second topic]. However, single-term queries usually give better results."
        # "Examples:\n"
        # "[urinary tract infection]\n"
        # "[Klebsiella]\n"
        # "[xylazine AND necrosis]\n"
        # "[pluripotent stem cells]"
    )
    # make a copy of the messages and add our request
    query_messages = []
    query_messages.extend(state['messages'])
    query_messages.append({
        'role': 'user',
        'content': query_compose_prompt
    })
    # get the plan
    response = completion(
        model=state['llm'],
        api_key=state['api_key'],
        base_url=state['base_url'],
        messages=query_messages,
        stream=False,
        max_tokens=256,
        stop=["]"],
        temperature=1.0,
        top_p=0.95
    )
    
    ai_msg = response['choices'][0]['message']
    print("Raw text:" + ai_msg['content'])
    new_query = ai_msg['content'].split('[')
    if len(new_query) == 2:
        new_query = new_query[1]
        print("PubMed query: " + new_query)
    else:
        print("Badly formatted response: " + ai_msg['content'])
        return {}
    # if query list hasn't been made, make it
    pubmed_queries = state.get('pubmed_queries')
    pubmed_query_pages = state.get('pubmed_query_pages')
    # if we haven't used this query before, add it
    if pubmed_queries is None:
        pubmed_queries = [ new_query ]
        pubmed_query_pages = [ 0 ]
    elif new_query in pubmed_queries: # if this query has been used before
        pubmed_query_pages[pubmed_queries.index(new_query)] += 1
    else: # query hasn't been used yet
        pubmed_queries = pubmed_queries + [ new_query ]
        pubmed_query_pages = pubmed_query_pages + [0]
    # return updates to state
    return {
        'pubmed_queries': pubmed_queries,
        'pubmed_query_pages': pubmed_query_pages,
        'current_pubmed_query': new_query
    }

def search_and_summarize(state: ResearchState):
    """
    Search PubMed using the most recent query and summarize each result.
    If the query has been used before, get the next 'page' of results and
    summarize those instead.
    """
    # prompt for summarizing a paper abstract
    abstract_summary_prompt = (
        "Please summarize the abstract of the following paper. Give a 2-3 sentence summary, "
        "focusing on the portions of the abstract that are relevant to your literature review topic.\n\n"
        "Title: {title}\n"
        "Publication date: {date}\n"
        "Abstract: {abstract}"
    )
    article_relevance_prompt = (
        "Is this article relevant? If it is relevant, write ##YES##. Then, in a single sentence, "
        "explain how this article is relevant to your literature review topic. If you think it is "
        "irrelevant, write ##NO##. Then explain why the article is irrelevant to your literature "
        "review. Be skeptical."
    )
    # get the search string
    current_query = state['current_pubmed_query']
    # get query page
    query_page = state['pubmed_query_pages'][state['pubmed_queries'].index(current_query)]
    # ask for the next 'page' of results if repeated query
    start_results = state['num_pubmed_results']*query_page
    # run the actual search
    search_tool = PubmedSearchTool()
    pubmed_results = search_tool.search_pubmed(query=current_query, start_results=start_results,
                                   max_results=state['num_pubmed_results'], sort='relevance')
    # if search returned no results, tell the agent that
    if (pubmed_results is None) or len(pubmed_results) == 0:
        print(f"[Warning] PubMed query {current_query} returned no results!")
        messages = state['messages']
        messages.append({
            'role': 'user',
            'content': (
                f"The PubMed query [{current_query}] did not return any results. "
                "Consider checking for errors or trying a less-specific query."
            )
        })
        messages.append({
            'role': 'assistant',
            'content': "Thank you for the feedback. I will try a different search approach next time."
        })
        return {
            'current_result_ids': [],
            # LLM with updated info on failed search
            'messages': messages
        }

    # record IDs of new articles so we can discuss them
    current_result_ids = []
    # get the article and summary dicts
    articles = state.get('articles')
    if articles is None:
        articles = {}
    article_summaries = state.get('article_summaries')
    if article_summaries is None:
        article_summaries = {}
    # grab LLM with current research state
    for result in pubmed_results:
        # check if we've already summarized this article
        if result['pubmed_id'] not in article_summaries.keys():
            # if not summarized
            # add to result list
            current_result_ids.append(result['pubmed_id'])
            # make a temporary branch for article summarization
            summary_msgs = []
            summary_msgs.extend(state['messages'])
            # summarize the abstract
            summary_msgs.append({
                'role': 'user',
                'content': abstract_summary_prompt.format(
                    title=result['title'],
                    date=result['date'],
                    abstract=result['abstract']
                )
            })
            response = completion(
                model=state['llm'],
                api_key=state['api_key'],
                base_url=state['base_url'],
                messages=summary_msgs,
                stream=False,
                max_tokens=512,
                # stop=["\n\n", "\n", "]"],
                temperature=1.0,
                top_p=0.95
            )
            summary = response['choices'][0]['message']
            summary_msgs.append({
                'role': 'assistant',
                'content': summary['content']
            })
            
            # evaluate article's relevance to the research topic
            summary_msgs.append({
                'role': 'user',
                'content': article_relevance_prompt
            })
            response = completion(
                model=state['llm'],
                api_key=state['api_key'],
                base_url=state['base_url'],
                messages=summary_msgs,
                stream=False,
                max_tokens=256,
                # stop=["\n\n", "\n", "]"],
                temperature=1.0,
                top_p=0.95
            )
            relevance = response['choices'][0]['message']
            summary_msgs.append({
                'role': 'assistant',
                'content': relevance['content']
            })
            # add summary and relevance to article dict
            result['summary'] = summary['content']
            result['relevance'] = relevance['content']
            # add to article set
            articles[result['pubmed_id']] = result
            # add to summaries
            article_summaries[result['pubmed_id']] = result['summary']
    # now return the updated article data
    return {
        "current_result_ids": current_result_ids,
        "articles": articles,
        "article_summaries": article_summaries
    }

def check_research_progress(state: ResearchState):
    """
    Decide whether the agent has collected enough information to answer the user's query.
    Write a few sentences explaining what information is missing or else why there is now
    enough information to respond to the user's query.
    """
    # article summary format
    article_format = (
        "[PMID {pubmed_id}] *{title}*\n"
        "**Summary:** {summary}\n"
        "**Relevance:** {relevance}\n\n"
    )
    prompt_search_result_summary = (
        "In a few sentences, please summarize any relevant information retrieved from your search results. "
        "Cite information using the source's PubMed ID surrounded by square brackets. Ignore any irrelevant information."
    )
    prompt_review_status = (
        "Please summarize the over-all status of your literature review thus far. Update your topic "
        "outline and indicate which topics require more information."
    )
    prompt_next_step = (
        "Based on your progress thus far, are there more topics you need to research? "
        "Are there gaps in your literature coverage that you need to fill? If you need "
        " to keep searching the literature, respond with ##YES##. If you have completely "
        "covered all research topics and do not need to do any more searches, respond with "
        "##NO##. If any of your research topics require any more information, respond with ##YES##."
    )
    # get the researcher notes
    researcher_notes = state['researcher_notes']
    # if there aren't any, start a list
    if researcher_notes is None:
        researcher_notes = []
        # if no results, skip to another search
    if len(state['current_result_ids']) == 0:
        researcher_notes.append("My search returned no results.")
        return {
            'mode': "search",
            'researcher_notes': researcher_notes
        }
    # get the LLM
    # ask for status of research first
    # we need to decide what to do next
    # we'll keep this off the main LLM chat
    status_check_msgs = []
    status_check_msgs.extend(state['messages'])
    # we left off with the LLM selecting a query
    # to keep conversation, have user 'ask' for search result summaries
    status_check_msgs.append({
        'role': 'user',
        'content': "What papers did your search turn up? Were they relevant?"
    })
    # now have LLM 'respond' with the summaries
    summary_txt = "I found the following papers:\n\n"
    for pmid in state['current_result_ids']:
        article = state['articles'][pmid]
        # only consider relevant articles here
        if "YES" in article['relevance']:
            summary_txt += article_format.format(
                pubmed_id = article['pubmed_id'],
                title = article['title'],
                summary = article['summary'],
                relevance = article['relevance']
            )
        else:
            print("Irrelevant article: " + article['relevance'])
    status_check_msgs.append({
        'role': 'assistant',
        'content': summary_txt
    })
    # summarize the latest round of search results
    status_check_msgs.append({
        'role': 'user',
        'content': prompt_search_result_summary
    })
    response = completion(
        model=state['llm'],
        api_key=state['api_key'],
        base_url=state['base_url'],
        messages=status_check_msgs,
        stream=False,
        max_tokens=2048,
        # stop=["\n\n", "\n", "]"],
        temperature=1.0,
        top_p=0.95
    )
    response = response['choices'][0]['message']
    result_summary = response['content']
    status_check_msgs.append({
        'role': 'assistant',
        'content': result_summary
    })
    
    # summarize over-all progress and update goals as needed
    status_check_msgs.append({
        'role': 'user',
        'content': prompt_review_status
    })
    response = completion(
        model=state['llm'],
        api_key=state['api_key'],
        base_url=state['base_url'],
        messages=status_check_msgs,
        stream=False,
        max_tokens=2048,
        # stop=["\n\n", "\n", "]"],
        temperature=1.0,
        top_p=0.95
    )
    response = response['choices'][0]['message']
    progress_summary = response['content']
    status_check_msgs.append({
        'role': 'assistant',
        'content': progress_summary
    })
    # ask LLM to select next step, returning either 'YES' or 'NO' for whether more research is needed
    status_check_msgs.append({
        'role': 'user',
        'content': prompt_next_step
    })
    response = completion(
        model=state['llm'],
        api_key=state['api_key'],
        base_url=state['base_url'],
        messages=status_check_msgs,
        stream=False,
        max_tokens=256,
        # stop=["\n\n", "\n", "]"],
        temperature=1.0,
        top_p=0.95
    )
    response = response['choices'][0]['message']
    next_step = response['content']
    status_check_msgs.append({
        'role': 'assistant',
        'content': next_step
    })
    if "##YES##" in next_step:
        mode = "search"
    else:
        mode = "report"
    # update main LLM conversation with just the progress summaries
    messages = state['messages']
    messages.append({
        'role': 'user',
        'content': prompt_search_result_summary
    })
    messages.append({
        'role': 'assistant',
        'content': result_summary
    })
    messages.append({
        'role': 'user',
        'content': prompt_review_status
    })
    messages.append({
        'role': 'assistant',
        'content': progress_summary
    })
    # add result summary to running list of notes
    researcher_notes.append(result_summary)
    # return results
    return {
        'messages': messages,
        'n_search_rounds': state['n_search_rounds'] + 1,
        'mode': mode,
        'researcher_notes': researcher_notes
    }

def route_researcher(state: ResearchState) -> str:
    """
    Determine the next step based on the research mode. Possible options
    are 'search' (run another PubMed search) or 'report' (write the final report).
    """
    # if we're past the maximum number of search rounds,
    # we will go to report mode regardless of whether we're finished searching
    if state['n_search_rounds'] > state['max_search_rounds']:
        return "report"
    # otherwise, we just respect the LLM's choice of modes
    return state['mode']

def compose_report(state: ResearchState):
    """
    Combine research results into a final report, including recommendations
    for important articles to read.
    """
    report_msgs = []
    # research completed, so standard prompt
    report_prompt = (
        "You are an experienced and capable graduate student. You are writing a thorough "
        "and comprehensive literature review for your thesis advisor. The literature reivew "
        "topic is:\n\n{topic}\n\n"
        "Here are the notes you took while searching the literature:\n\n{notes}\n\n"
        "You found the following relevant papers:\n\n{papers}\n\n"
        "You will take this information and synthesize it into a well-written literature review."
    )

    article_format = (
        "[PMID {pubmed_id}] *{title}*\n"
        "**Summary:** {summary}\n"
        "**Relevance:** {relevance}\n\n"
    )

    # if mode is still 'search' and maximum search rounds exceeded, add a
    # prompt here to acknowledge incomplete search and point out gaps when
    # writing the report
    if state['mode'] == "search" and state['n_search_rounds'] > state['max_search_rounds']:
        report_prompt += (
            "\n\nYou ran out of time to finish searching the literature, so there are "
            "some gaps in your information. Be sure to point out where more information "
            "is needed."
        )
    
    # initial outline prompt
    outline_prompt = (
        "Start by writing an outline of the review."
    )

    # assemble the research notes
    notes = "\n\n".join(state['researcher_notes'])

    # assemble the list of relevant articles
    summary_txt = ""
    for pmid in state['current_result_ids']:
        article = state['articles'][pmid]
        # only consider relevant articles here
        if "YES" in article['relevance']:
            summary_txt += article_format.format(
                pubmed_id = article['pubmed_id'],
                title = article['title'],
                summary = article['summary'],
                relevance = article['relevance']
            )
    
    report_msgs.append({
        'role': 'system',
        'content': report_prompt.format(
            topic=state['user_query'],
            notes=notes,
            papers=summary_txt
        )
    })

    # user asks for an outline
    report_msgs.append({
        'role': 'user',
        'content': outline_prompt
    })
    response = completion(
        model=state['llm'],
        api_key=state['api_key'],
        base_url=state['base_url'],
        messages=report_msgs,
        stream=False,
        max_tokens=2048,
        # stop=["\n\n", "\n", "]"],
        temperature=1.0,
        top_p=0.95
    )
    response = response['choices'][0]['message']
    report_outline = response['content']
    return {
        "final_report": report_outline,
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
        with tracer.start_as_current_span("Invoke ResearchAgent") as session_span:
            final_result = self.agent_graph.invoke(
                input=agent_state,
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
        self.agent_graph = StateGraph(ResearchState)

        # define nodes
        self.agent_graph.add_node("start_research", start_research)
        self.agent_graph.add_node("compose_pubmed_query", compose_pubmed_query)
        self.agent_graph.add_node("search_and_summarize", search_and_summarize)
        self.agent_graph.add_node("check_research_progress", check_research_progress)
        self.agent_graph.add_node("compose_report", compose_report)
        
        # define edges
        self.agent_graph.add_edge(START, "start_research")
        self.agent_graph.add_edge("start_research", "compose_pubmed_query")
        self.agent_graph.add_edge("compose_pubmed_query", "search_and_summarize")
        self.agent_graph.add_edge("search_and_summarize", "check_research_progress")
        # conditional edge for search mode or report mode
        self.agent_graph.add_conditional_edges(
            "check_research_progress",
            route_researcher,
            {
                "search": "compose_pubmed_query",
                "report": "compose_report"
            }
        )
        self.agent_graph.add_edge("compose_report", END)
        
        # compile graph
        self.agent_graph = self.agent_graph.compile()