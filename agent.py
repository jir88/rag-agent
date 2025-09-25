from typing import TypedDict, List, Dict, Any, Optional, Tuple
from pydantic import BaseModel
from litellm import completion
from langgraph.graph import StateGraph, START, END

####################
# Assistant prompts
####################
PLANNER_MESSAGE = """You are a task planner. You will be given a task. Your job is to think step by step and enumerate the steps to complete a given task, using the provided context to guide you.
    You will not execute the steps yourself, but provide the steps to a helper who will execute them. Make sure each step consists of a single operation, not a series of operations. The helper has the following capabilities:
    1. Search through a collection of documents provided by the user. These are the user's own documents and will likely not have latest news or other information you can find on the internet.
    2. Synthesize, summarize and classify the information received.
    3. Search the internet
    The plan may have as few or as many steps as is necessary to accomplish the given task.

    You may direct the helper to use any of the capabilties that it has, but you do not need to use all of them if they are not required to complete the task.
    For example, if the task requires knowledge that is specific to the user, you may choose to include a step that searches through the user's documents. However, if the task only requires information that is available on the internet, you may choose to include a step that searches the internet and omit document searching.

    You must format the plan steps as a JSON object. Use the following format:

    **OUTPUT FORMAT (JSON):**
    {
        "steps": "A list of strings containing each step in the plan. Each entry in the list should be a single step."
    }
    """

ASSISTANT_PROMPT = """You are an AI assistant.
    When you receive a message, figure out a solution and provide a final answer. The message will be accompanied with contextual information. Use the contextual information to help you provide a solution.
    Make sure to provide a thorough answer that directly addresses the message you received.
    If tool calls are used, **include ALL retrieved details** from the tool results.
    **DO NOT summarize** multiple sources into a single explanation—**instead, cite each source individually**.
    When citing sources returned from tool calls, you must always provide the source URL or the source document name.
    When you are using knowledge and web search tools to complete the instruction, answer the instruction only using the results from the search; do no supplement with your own knowledge.
    Be persistent in finding the information you need before giving up.
    If the task is able to be accomplished without using tools, then do not make any tool calls.
    When you have accomplished the instruction posed to you, you will reply with the text: ##ANSWER## - followed with an answer.
    Important: If you are unable to accomplish the task, whether it's because you could not retrieve sufficient data, or any other reason, reply only with ##TERMINATE##.

    # Tool Use
    You have access to the following tools. Only use these available tools and do not attempt to use anything not listed - this will cause an error.
    Respond in the format: <|tool_call|>{"name": function name, "arguments": dictionary of argument name and its value}. Do not use variables.
    Only call one tool at a time.
    When you are using knowledge and web search tools to complete the instruction, answer the instruction only using the results from the search; do no supplement with your own knowledge.
    Never answer the instruction using links to URLs that were not discovered during the use of your search tools. Only respond with document links and URLs that your tools returned to you.
    Also make sure to provide the URL for the page you are using as your source or the document name.
    """

GOAL_JUDGE_PROMPT = """You are a judge. Your job is to carefully inspect whether a stated goal has been **fully met**, based on all of the requirements of the provided goal, the plans drafted to achieve it, and the information gathered so far.

## **STRICT INSTRUCTIONS**  
- You **must provide exactly one response**—either **##YES##** or **##NOT YET##**—followed by a brief explanation.  
- If **any** part of the goal remains unfulfilled, respond with **##NOT YET##**.  
- If and only if **every single requirement** has been met, respond with **##YES##**.  
- Your explanation **must be concise (1-2 sentences)** and clearly state the reason for your decision.  
- **Do NOT attempt to fulfill the goal yourself.**  
- If the goal involves gathering specific information (e.g., fetching internet articles) and this has **not** been done, respond with **##NOT YET##**.  

    **OUTPUT FORMAT:**  
    ```
    ##YES## or ##NOT YET##      
    Explanation: [Brief reason why this conclusion was reached]
    ```

    **INPUT FORMAT (JSON):**
    ```
    {
        "Goal": "The ultimate goal/instruction to be fully fulfilled, along with any accompanying images that may provide further context.",
        "Media Description": "If the user provided an image to supplement their instruction, a description of the image's content."
        "Plan": "The plan to achieve the goal, including any sub-goals or tasks that need to be completed.",
        "Information Gathered": "The information collected so far in pursuit of fulfilling the goal."
    }
    ```

## **REMEMBER:**  
- **Provide only ONE response**: either **##YES##** or **##NOT YET##**.  
- The explanation must be **concise**—no more than **1-2 sentences**.  
- **If even a small part of the goal is unfulfilled, reply with ##NOT YET##.**  
    """

REFLECTION_ASSISTANT_PROMPT = """You are a strategic planner focused on executing sequential steps to achieve a given goal. You will receive data in JSON format containing the current state of the plan and its progress. Your task is to determine the single next step, ensuring it aligns with the overall goal and builds upon the previous steps.

JSON Structure:
{
    "Goal": The original objective from the user,
    "Media Description": A textual description of any associated image,
    "Plan": An array outlining every planned step,
    "Last Step": The most recent action taken,
    "Last Step Output": The result of the last step, indicating success or failure,
    "Steps Taken": A chronological list of executed steps.
}

Guidelines:
1. If the last step output is ##NO##, reassess and refine the instruction to avoid repeating past mistakes. Provide a single, revised instruction for the next step.
2. If the last step output is ##YES##, proceed to the next logical step in the plan.
3. Use 'Last Step', 'Last Output', and 'Steps Taken' for context when deciding on the next action.

Restrictions:
1. Do not attempt to resolve the problem independently; only provide instructions for the subsequent agent's actions.
2. Limit your response to a single step or instruction.

Example of a single instruction:
- "Analyze the dataset for missing values and report their percentage."
    """

STEP_CRITIC_SYSTEM_PROMPT = "You are a quality control agent. Your job is to determine whether the other agents are doing their jobs correctly."

STEP_CRITIC_PROMPT = """The previous instruction was {last_step} \nThe following is the output of that instruction.
    if the output of the instruction completely satisfies the instruction, then reply with ##YES##.
    For example, if the instruction is to list companies that use AI, then the output contains a list of companies that use AI.
    If the output contains the phrase 'I'm sorry but...' then it is likely not fulfilling the instruction. \n
    If the output of the instruction does not properly satisfy the instruction, then reply with ##NO## and the reason why.
    For example, if the instruction was to list companies that use AI but the output does not contain a list of companies, or states that a list of companies is not available, then the output did not properly satisfy the instruction.
    If it does not satisfy the instruction, please think about what went wrong with the previous instruction and give me an explanation along with the text ##NO##. \n
    Previous step output: \n {last_output}"""

class AgentPlan(BaseModel):
    """Pydantic model defining how LLM should format steps in its research plan."""
    steps: list[str]

class AgentState(TypedDict):
    # the LiteLLM path to the inference client we're using to talk to an LLM
    llm: str
    # optional API key for inference client
    api_key: Optional[str] = None
    # optional URL where inference client is located
    base_url: Optional[str] = None
    # # system prompt including tools the agent is allowed to use
    # system_prompt: str
    # the user's question that we're trying to answer
    user_query: Optional[str]
    # # path to a file related to the query
    # user_file: Optional[str] = None
    # # message history thus far
    # messages: List[Dict[str, str]]
    # current plan
    plan: List[str]
    # number of search rounds we've tried
    n_search_rounds:int = 0
    # maximum number of search rounds
    max_search_rounds: int = 10
    # the previously run step
    last_step: Optional[str]
    # This variable tracks the output of previous successful steps as context for executing the next step
    answer_output: Tuple[List[str]]
    steps_taken: List[str]  # A list of steps already executed
    last_output: str  # Output of the single previous step gets put here
    # # the latest code block generated by the agent
    # current_code_block: Optional[str]
    # # python sandbox
    # sandbox: LocalPythonExecutor
    # have we gotten a final answer?
    has_final_answer:bool = False
    # what is the final answer?
    final_answer:Optional[str]

# ------------------ Workflow Steps --------------

def make_plan(state:AgentState):
    """
    Creates the initial high-level plan, once, in the beginning of the workflow. 
    For example, if a user asks, “What are comparable open source projects to the ones my team
    is using?” then the agent will put together a step-by-step plan that may look something like 
    this: “1. Search team documents for open source technologies. 2. Search the web for similar 
    open source projects to the ones found in step 1.” If any of these steps fail or provide 
    insufficient results, the steps can be later adapted by the Reflection Agent.
    """
    # send user query to the planner
    planning_messages = [
        # system message defining the planner agent
        {
            'role': 'system',
            'content': PLANNER_MESSAGE
        },
        # the actual user query
        {
            'role': 'user',
            'content': state['user_query']
        }
    ]

    # get the plan
    # TODO: force JSON output
    response = completion(
        model=state['llm'],
        api_key=state['api_key'],
        base_url=state['base_url'],
        messages=planning_messages,
        response_format=AgentPlan,
        # stop=["Code:", "```", "\n\n"],
        stream=False,
        max_tokens=1024,
        temperature=1.0,
        top_p=0.95
    )
    
    ai_msg = response['choices'][0]['message']
    print("Raw text:" + ai_msg['content'])
    plan = AgentPlan.model_validate_json(ai_msg['content'])
    print("Initial plan:\n\n" + str(plan))
    
    return {
        "plan": plan
    }

def check_agent_progress(state:AgentState):
    """
    If the agent reports that it has a final answer or the number of search rounds is too high,
    respond. Otherwise, do another round of thinking and code writing.
    """
    if state['has_final_answer'] or (state['n_search_rounds'] > state['max_search_rounds']):
        return 'respond'
    else:
        return 'continue'

def do_research_step(state:AgentState):
    """
    The Research Assistant is the workhorse of the system. It takes in and 
    executes instructions such as, “Search team documents for open source technologies.” For 
    step 1 of the plan, it uses the initial instruction from the Planner Agent. For subsequent 
    steps, it also receives curated context from the outcomes of previous steps. For example, 
    if tasked with “Search the web for similar open source projects,” it will also receive the 
    output from the previous document search step. Depending on the instruction, the Research 
    Assistant can use tools like web search or document search, or both, to fulfill its task.
    """
    current_step = state['plan'].steps[state['n_search_rounds']]
    print("Starting step: " + current_step)

    return {
        'final_answer': "42"
    }

def summarize_research_result(state:AgentState):
    """
    The Summarizer Agent condenses the Research Assistant’s findings into a 
    concise, relevant response. For example, if the Research Assistant finds detailed meeting 
    notes stating, “We discussed the release of Tool X that uses Tool Y underneath,” then the 
    Summarizer Agent extracts only the relevant snippets such as, "Tool Y is being used," and 
    reformulates it to directly answer the original instruction. This may seem like a small 
    detail, but it can help give higher quality results and keep the model on task, especially 
    as one step builds upon the output of another step.
    """
    pass

def evaluate_step_success(state:AgentState):
    """
    Use critic agent to decide whether or not the current step was completed successfully.
    """
    reflection_message = state['last_step']
    # format the prompt for the critic
    critic_prompt = STEP_CRITIC_PROMPT.format(
        last_step=state['last_step'],
        context=answer_output,
    )
    # Ask the critic if the previous step was properly accomplished
    critic_msgs = [
        {
            'role': 'system',
            'content': STEP_CRITIC_SYSTEM_PROMPT
        }
    ]
    # output = await user_proxy.a_initiate_chat(
    #     recipient=step_critic,
    #     max_turns=1,
    #     message=STEP_CRITIC_PROMPT.format(
    #         last_step=last_step,
    #         context=answer_output,
    #         last_output=last_output,
    #     ),
    # )
    
    was_job_accomplished = output.chat_history[-1]["content"]

def check_agent_progress(state:AgentState):
    """
    If the agent reports that it has a final answer or the number of search rounds is too high,
    respond. Otherwise, do another round of thinking and code writing.
    """
    if state['has_final_answer'] or (state['n_search_rounds'] > state['max_search_rounds']):
        return 'respond'
    else:
        return 'continue'

class RagAgent:
    """
    A multi-agent deep research system based on IBM's RAG agent implementation.

    Architecture:
        Planning agent: Creates the initial high-level plan, once, in the beginning of the workflow. 
            For example, if a user asks, “What are comparable open source projects to the ones my team
            is using?” then the agent will put together a step-by-step plan that may look something like 
            this: “1. Search team documents for open source technologies. 2. Search the web for similar 
            open source projects to the ones found in step 1.” If any of these steps fail or provide 
            insufficient results, the steps can be later adapted by the Reflection Agent.

        Research assistant: The Research Assistant is the workhorse of the system. It takes in and 
            executes instructions such as, “Search team documents for open source technologies.” For 
            step 1 of the plan, it uses the initial instruction from the Planner Agent. For subsequent 
            steps, it also receives curated context from the outcomes of previous steps. For example, 
            if tasked with “Search the web for similar open source projects,” it will also receive the 
            output from the previous document search step. Depending on the instruction, the Research 
            Assistant can use tools like web search or document search, or both, to fulfill its task.

        Summarizer agent: The Summarizer Agent condenses the Research Assistant’s findings into a 
            concise, relevant response. For example, if the Research Assistant finds detailed meeting 
            notes stating, “We discussed the release of Tool X that uses Tool Y underneath,” then the 
            Summarizer Agent extracts only the relevant snippets such as, "Tool Y is being used," and 
            reformulates it to directly answer the original instruction. This may seem like a small 
            detail, but it can help give higher quality results and keep the model on task, especially 
            as one step builds upon the output of another step.

        Critic agent: The Critic Agent is responsible for deciding whether the output of the previous 
            step satisfactorily fulfilled the instruction it was given. It receives two pieces of 
            information: the single step instruction that was just executed and the output of that 
            instruction from Summarizer Agent. Having a Critic Agent weigh in on the conversation 
            brings clarity around whether the goal was achieved, which is needed for the planning 
            of the next step.

        Reflection agent: The reflection agent is our executive decision maker. It decides what step
            to take next, whether that is encroaching onto the next planned step, pivoting course to 
            make up for mishaps or giving the thumbs up that the goal has been completed. Much like a 
            real-life CEO, it performs its best decision making when it has a clear goal in mind and 
            is presented with concise findings on the progress that has or has not been made to reach 
            that goal. The output of the Reflection Agent is either the next step to take or the 
            instructions to terminate if the goal has been reached. We present the Reflection Agent 
            with the following items:
                - The goal.
                - The original plan.
                - The last step executed.
                - The output of the Summarizer and Critic Agents from the last step.
                - A concise sequence of previously executed instructions (just the instructions, not their output).
            Presenting these items in a structured format makes it clear to our decision maker what 
            has been done so that it can decide what needs to happen next.

        Report generator: Once the goal is achieved, the Report Generator synthesizes all findings 
            into a cohesive output that directly answers the original query. While each step in the 
            process generates targeted outputs, the Report Generator ties everything together into a 
            final report.
    """

    def __init__(
        self, llm:str, tools:Dict[str, Any]={}, py_imports:List[str]=[],
        api_key:str = None, base_url:str = None
    ):
        """
        Create a new agent with a set of tools.

        Args:
            llm (str): LiteLLM identifier of the model to use for LLM inference
            tools (dict): A dict mapping custom tool names to tool objects
            py_imports (list): A list of python packages the agent is allowed to import
            api_key (str): Optional API key if the LLM service requires it
            base_url (str): Optional custom URL where LLM service is located
        """
        self.llm = llm
        self.api_key = api_key
        self.base_url = base_url
        # create system prompt
        # self.system_prompt = initialize_system_prompt(
        #     sys_prompt=SMOL_CODE_PROMPT,
        #     tools=tools,
        #     auth_imp=py_imports)
        # # create python sandbox with tools
        # self.py_interp = LocalPythonExecutor(
        #     additional_authorized_imports=py_imports, 
        #     max_print_outputs_length=None
        #     )
        # manually add base python tools
        # self.py_interp.static_tools = BASE_PYTHON_TOOLS
        # add the final_answer tool and any custom tools
        # tools["final_answer"] = FinalAnswerTool()
        # self.py_interp.send_tools(tools=tools)
        # build the agent graph
        self._initialize_agent_graph()

    def submit_question(self, question:str, file_name:str = None):
        """
        Submit a question for this agent to answer.

        Args:
            question (str): The question text.
            file_name (str): Path to a file to use when answering this question.
        
        Return:
            Answer to question in text form.
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
            # no answer outputs yet
            'answer_output': (),
            # no steps executed yet
            'steps_taken': [],
            # no output yet
            'last_output': "",
        }
        final_result = self.agent_graph.invoke(agent_state)
        print(final_result)
        return final_result

    def _initialize_agent_graph(self):
        """
        Assemble the agent graph and compile it.
        """
        # construct graph object
        self.agent_graph = StateGraph(AgentState)

        # define nodes
        self.agent_graph.add_node("make_plan", make_plan)
        self.agent_graph.add_node("do_research_step", do_research_step)
        # self.agent_graph.add_node("make_observation", make_observation)
        # self.agent_graph.add_node("think", think)
        # self.agent_graph.add_node("respond", respond)
        
        # define edges
        self.agent_graph.add_edge(START, "make_plan")
        self.agent_graph.add_edge("make_plan", "do_research_step")
        self.agent_graph.add_edge("do_research_step", END)
        # self.agent_graph.add_edge("make_plan", "write_code")
        # self.agent_graph.add_edge("write_code", "make_observation")
        # self.agent_graph.add_conditional_edges(
        #     "make_observation",
        #     check_agent_progress,
        #     {
        #         "continue": "think",
        #         "respond": "respond"
        #     }
        # )
        # self.agent_graph.add_edge("think", "write_code")
        # self.agent_graph.add_edge("respond", END)

        # compile graph
        self.agent_graph = self.agent_graph.compile()