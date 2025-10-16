import tools
from agent import RagAgent

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
    agent = RagAgent(
        llm=model, 
        tools={
            "search_pubmed": tools.PubmedSearchTool(),
            # "web_search": DuckDuckGoSearchTool(max_results=10),
            # "visit_webpage": VisitWebpageTool(),
            # "wikipedia_search": WikipediaSearchTool(max_results=4)
        },
        api_key=api_key,
        base_url=base_url
    )

    result = agent.submit_question("Search online for recent research on bile acid metabolism and write a detailed report describing any significant findings.")
    print(result['final_answer'])