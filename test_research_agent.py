import json
from researcher import ResearchAgent

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
    agent = ResearchAgent(
        llm=model,
        api_key=api_key,
        base_url=base_url
    )

    query = (
        "Summarize recent research on bile acid metabolism. Perform several initial searches with general keywords "
        "before focusing on particular areas of active research with follow-up queries. Look for both host and "
        "microbiome interactions with bile acids and the effects of these interactions on disease."
    )
    result = agent.submit_question(question=query)
    with open("output_research_state.json", mode='w') as fp:
        json.dump(result, fp=fp, indent=2)
    print(json.dumps(result, indent=2))