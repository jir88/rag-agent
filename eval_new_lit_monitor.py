import datetime
import json
import sys
import pandas as pd
import new_lit_monitor as nlm

from pathlib import Path
from typing import List, Dict, Any
from langgraph.graph import StateGraph, START, END

from opentelemetry import trace
from opentelemetry import trace as trace_api  # SDK for creating traces

# grab the actual tracer being used
tracer = trace.get_tracer(__name__)

# TODO: set up evaluation agent using subset of steps, skipping the initial PubMed search

class EvalLitMonitor:
    """
    Workflow powered by LiteLLM that evaluates LitMonitor agent performance by testing against
    a fixed set of articles that have been hand-evaluated for relevance to the associated topic.
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
    
    # TODO: change to evaluate_on_articles
    # replace prior_pmids with list of articles to evaluate on
    # what format are articles stored in?
    # push them into 'new_articles' state variable
    # run agent
    # profit
    def evaluate_on_articles(
        self, topic_description:str,
        article_list:List[Dict[str, Any]] = []
        ):
        """
        Run the monitor agent for a given topic and search term.

        Args:
            topic_description (str): Description of the research topic to help the agent 
                decide whether articles are relevant.
            article_list (List[Dict[str, Any]]): A list of the articles we want to evaluate
                the agent on. Article dictionaries should be formatted identically to those 
                produced by the do_search node in the main agent.
        
        Returns:
            The results of the agent evaluation run in a dict.
        """
        # build agent state
        state = {
            'llm': self.llm,
            'api_key': self.api_key,
            'base_url': self.base_url,
            'topic_description': topic_description,
            'search_terms': "",
            'article_evaluations': [],
            'new_articles': article_list,
            'num_pubmed_results': -1
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
        self.agent_graph = StateGraph(nlm.LitMonitorState)

        # define nodes
        self.agent_graph.add_node("eval_all_papers", nlm.eval_all_papers)
        self.agent_graph.add_node("collate_evals", nlm.collate_evals)
        self.agent_graph.add_node("save_results", nlm.save_results)
        
        # define edges
        self.agent_graph.add_edge(START, "eval_all_papers")
        self.agent_graph.add_edge("eval_all_papers", "collate_evals")
        self.agent_graph.add_edge("collate_evals", "save_results")
        self.agent_graph.add_edge("save_results", END)
        
        # compile graph
        self.agent_graph = self.agent_graph.compile()

def main():
    """
    Main function that loads a set of test articles and evaluates them against lit monitor configurations.
    """
    # make sure we have at least one argument
    if len(sys.argv) < 2:
        print("Usage: python eval_new_lit_monitor.py <eval_file.csv>")
        sys.exit(1)

    # try loading the eval configuration file
    input_path = Path(sys.argv[1])
    eval_dir = input_path.parent

    if not input_path.exists():
        print(f"Error: Topic file not found: {input_path}")
        sys.exit(1)

    # Read the evaluation configuration file
    try:
        validation_queries = pd.read_csv(input_path)
    except Exception as e:
        print(f"Error reading topic file: {e}")
        sys.exit(1)
    
    # for each evaluation in the configuration file
    for row in validation_queries.itertuples():
        print(row)
        # not sure how to handle API keys, but not relevant for me...
        api_key="sk-fake"

        agent = EvalLitMonitor(
            llm=row.model,
            api_key=api_key,
            base_url=row.base_url
        )

        # read evaluation file
        # file is relative to the validation queries table
        qr_path = eval_dir.joinpath(row.query_results)
        article_table = pd.read_csv(qr_path)
        # run current monitor on these files
        # don't go row by row, just build the whole column and send it
        # grab the article data columns only
        article_data_only = article_table[["pubmed_id", "doi", "title", "date", "authors", "source", "abstract"]]
        # convert to a list of dicts
        article_dicts = article_data_only.to_dict('records')
        # feed it to the eval monitor
        result = agent.evaluate_on_articles(
            topic_description=row.query,
            article_list=article_dicts
        )
        # put timestamp on outputs
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # save whole result dict
        with open("evals/eval_results_" + row.query_results + "_" + timestamp + ".json", mode='w') as fp:
            json.dump(result, fp=fp, indent=2)
        # we can convert the article values straight into a DataFrame and write it to CSV for evals
        new_articles = result['new_articles']
        df = pd.DataFrame(new_articles)
        # add LLM metadata
        df['model'] = row.model
        df['base_url'] = row.base_url
        df.to_csv("evals/eval_results_" + row.query_results + "_" + timestamp + ".csv")
        print(json.dumps(result, indent=2))
    # end eval file loop


if __name__ == "__main__":
    main()