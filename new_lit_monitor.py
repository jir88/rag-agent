import argparse
import sys
import datetime
# import pandas as pd

from openai import OpenAI
from pathlib import Path
from typing import List, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

from monitor import Article,LitMonitorState
from tools import PubmedSearchTool

def do_search(state: LitMonitorState) -> Command[Literal["eval_all_papers", "collate_evals"]]:
    """
    Do the PubMed search and grab metadata for any articles we haven't seen before.
    """
    # get the search string
    current_query = state.search_terms
    # how far back do we need to search?
    max_results = state.num_pubmed_results + len(state.prior_pmids)
    # run the actual search
    search_tool = PubmedSearchTool()
    pubmed_results = search_tool.search_pubmed(
        query=current_query, 
        start_results=0,
        max_results=max_results,
        sort='pub_date')

    # get any new articles
    new_results = [res for res in pubmed_results if res['pubmed_id'] not in state.prior_pmids]
    # convert dict results into Article objects
    new_objs = [Article.model_validate(res) for res in pubmed_results]

    if len(new_results) == 0:
        return Command(
            update={
                'new_articles': []
            },
            goto="collate_evals"
        )
    return Command(
        update={
            'new_articles': new_objs
        },
        goto="eval_all_papers"
    )

def eval_all_papers(state:LitMonitorState) -> LitMonitorState:
    """
    Evaluate all the new papers for relevance.
    """
    new_articles = state.new_articles
    for article in new_articles:
        # make conversation
        relevance_msgs = [
            {
                'role': 'system',
                'content': state.agent_system_prompt.format(
                    topic=state.topic_description
                )
            },
            {
                'role': 'user',
                'content': state.article_relevance_prompt.format(
                    title=article.title,
                    date=article.date,
                    abstract=article.abstract
                )
            }
        ]
        # try a couple times in case LLM doesn't do it right
        attempts = 0
        max_attempts = 3
        is_relevant = None

        while attempts < max_attempts:
            response = state.client.chat.completions.create(
                model=state.llm,
                messages=relevance_msgs,
                stream=False,
                # shove all sampling parameters through this mechanism to avoid manually
                # specifying the canonical OpenAI ones
                extra_body=state.sampling_params
            )
            rel_response = response.choices[0].message.content
            print(rel_response)

            # determine if LLM flags as relevant or not
            if "YES" in rel_response:
                is_relevant = True
                attempts = max_attempts # end loop
            elif "NO" in rel_response:
                is_relevant = False
                attempts = max_attempts # end loop
            else:
                print("LLM response did not contain NO or YES! Retrying...")
                attempts += 1

        # record evaluation, relevance flag will be None if we failed
        article.evaluation = rel_response
        article.is_relevant = is_relevant
    
    # update state with article list, now with relevance added
    return {
        'new_articles': new_articles
    }

def collate_evals(state:LitMonitorState) -> dict:
    print(str(state))
    return {}

def save_results(state:LitMonitorState):
    return {}

class LitMonitor:
    """
    Workflow powered by LiteLLM that gets new papers from PubMed and decides
    whether they are relevant to the user's interests.
    """

    def __init__(self, llm:str, api_key:str = None, base_url:str = None, sampling_params:dict = None):
        """
        Create a new monitor workflow with a back-end LLM.

        Args:
            llm (str): LiteLLM identifier of the model to use for LLM inference
            api_key (str): Optional API key if the LLM service requires it
            base_url (str): Optional custom URL where LLM service is located
            sampling_params (dict): OpenAI styled dict of sampling parameters to use with the LLM
        """
        self.llm = llm
        self.api_key = api_key
        self.base_url = base_url
        self.sampling_params = sampling_params

        # default parameters if none provided
        if self.sampling_params is None:
            # values have to be strings
            self.sampling_params = {
                "temperature": "1.0",
                "top_p": "0.95"
            }

        # create OpenAI API client
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=1200,
            max_retries=10
        )

        # build the agent graph
        self._initialize_agent_graph()
    
    def check_search(
        self, 
        system_prompt:str, article_relevance_prompt:str,
        topic_description:str, search_terms:str,
        max_results:int=25,
        prior_pmids:List[str] = []
        ) -> LitMonitorState:
        """
        Run the monitor agent for a given topic and search term.

        Args:
            system_prompt (str): The LLM system prompt that sets up the article evaluation 
                scenario for the LLM. Must include f-string style slot for {topic}.
            article_relevance_prompt (str): The skeleton prompt into which the article data 
                is injected. Should end by prompting the LLM to evaluate the article relevance 
                and output either ##YES## or ##NO## in response. Must include f-string style 
                slots for {title}, {date}, and {abstract}.
            topic_description (str): Description of the research topic to help the agent 
                decide whether articles are relevant.
            search_terms (str): The search to use in PubMed search syntax.
            max_results (int): The maximum number of search results to evaluate.
            prior_pmids (List[str]): Optional list of PMIDs we've seen before, 
                to avoid checking the same articles repeatedly.
        
        Returns:
            The results of the agent run in a dict.
        """
        # build agent state
        state = {
            'llm': self.llm,
            'api_key': self.api_key,
            'base_url': self.base_url,
            'sampling_params': self.sampling_params,
            'client': self.client,
            'agent_system_prompt': system_prompt,
            'article_relevance_prompt': article_relevance_prompt,
            'topic_description': topic_description,
            'search_terms': search_terms,
            'prior_pmids': prior_pmids,
            'article_evaluations': [],
            'num_pubmed_results': max_results
        }
        final_result = self.agent_graph.invoke(
            input=state,
            config={ "recursion_limit": 100 }
        )
        output_model = LitMonitorState(**final_result)
        print("Agent finished!")
        return output_model

    def _initialize_agent_graph(self):
        """
        Assemble the agent graph and compile it.
        """
        # construct graph object
        self.agent_graph = StateGraph(LitMonitorState)

        # define nodes
        self.agent_graph.add_node("do_search", do_search)
        self.agent_graph.add_node("eval_all_papers", eval_all_papers)
        self.agent_graph.add_node("collate_evals", collate_evals)
        self.agent_graph.add_node("save_results", save_results)
        
        # define edges
        self.agent_graph.add_edge(START, "do_search")
        # self.agent_graph.add_edge("do_search", "eval_all_papers")
        self.agent_graph.add_edge("eval_all_papers", "collate_evals")
        self.agent_graph.add_edge("collate_evals", "save_results")
        self.agent_graph.add_edge("save_results", END)
        
        # compile graph
        self.agent_graph = self.agent_graph.compile()

def parse_args() -> argparse.Namespace:
    """
    Set up command line argument parsing.
    """
    parser = argparse.ArgumentParser(description=(
        "Check the relevance of the most recent papers matching a PubMed search. "
        "Default behavior is to ignore any list of previously-seen articles in the input file and "
        "to output results to a new file with the same name as the "
        "input file plus a timestamp. If --update is used, previously-seen articles in the input file "
        "are NOT reevaluated, and the output results will be combined "
        "with the input file and written to that file. The --output option allows one to specify a name "
        "for the output file, regardless of whether the --update flag is set."
    ))
    parser.add_argument("filename", type=Path, help="Path to the configuration file to use")
    parser.add_argument("-U", "--update", action="store_true",
                        help="Update the previously-run search described in the configuration file.")
    parser.add_argument("-O", "--output", type=Path, action="store", help="Path to the output file to store results.")
    parser.add_argument("-K", "--key", type=str, action="store", help="API key to use with the LLM, if needed.")
    return parser.parse_args()

def main():
    """
    The main method for running the literature monitor from the command line.
    Takes one unnamed argument, the name of the topic file to use.
    """
    # automatically parse the arguments
    args = parse_args()

    # try loading the topic file
    input_path = args.filename
    
    if not input_path.exists():
        raise FileNotFoundError(f"Error: Topic file not found: {input_path}")
    
    # Read the topic file
    try:
        input_settings = LitMonitorState.model_validate_json(input_path.read_text())
    except Exception as e:
        print(f"Error reading topic file: {e}")
        sys.exit(1)

    # get API key, if provided
    if args.key is not None:
        print("Custom key provided!")
        api_key = args.key
    else:
        api_key = "sk-fake"

    agent = LitMonitor(
        llm=input_settings.llm,
        api_key=api_key,
        base_url=input_settings.base_url,
        sampling_params=input_settings.sampling_params
    )

    # if we are updating, need to provide the list of prior results
    if args.update:
        # grab list of prior PMIDs and put it in a set
        prior_pmids = set(input_settings.prior_pmids)
        # add the list of new articles
        for article in input_settings.new_articles:
            prior_pmids.add(article['pubmed_id'])
        # convert to a list
        prior_pmids = list(prior_pmids)
    else:
        prior_pmids = []
    print("Prior PMIDs: " + str(prior_pmids))
    
    print(f"Running agent with query:\n\n{input_settings.topic_description}\n\nSearch term :{input_settings.search_terms}\n\n")
    result = agent.check_search(
        system_prompt=input_settings.agent_system_prompt,
        article_relevance_prompt=input_settings.article_relevance_prompt,
        topic_description=input_settings.topic_description, 
        search_terms=input_settings.search_terms,
        max_results=input_settings.num_pubmed_results,
        prior_pmids=prior_pmids
    )

    # if we're updating, add the new articles to the existing ones
    if args.update:
        updated_list = input_settings.new_articles.extend(result.new_articles)
        result.new_articles = updated_list
        result.prior_pmids = input_settings.prior_pmids

    # if output file is specified, use that
    if args.output is not None:
        output_file_name = args.output
    elif args.update:
        # use the file we're updating
        output_file_name = input_path
    else:
        # put timestamp on input file name
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_file_name = input_path.parent.joinpath(input_path.stem + "_" + timestamp + ".json")
    # save whole result dict
    with open(output_file_name, mode='w') as fp:
        fp.write(result.model_dump_json(indent=2, ensure_ascii=True))

if __name__ == "__main__":
    main()