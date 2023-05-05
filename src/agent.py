import os
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain, PromptTemplate
from langchain.tools import Tool
from langchain.agents import  StructuredChatAgent, AgentExecutor
from langchain.utilities import WikipediaAPIWrapper, WolframAlphaAPIWrapper, SerpAPIWrapper
from src.qa import QuestionAnswerer
from src.tools import WorldAnvilTool, CreativeTool, SearchInput, creative_template

prefix = """Respond to the human as helpfully and accurately as possible. You have access to the following tools:"""

suffix = """Begin! Reminder to ALWAYS respond with a valid json blob of a single action. Use tools if necessary. Respond directly if appropriate. Format is Thought, Action:```$JSON_BLOB```then Observation:.
Thought:"""


class Agent:
    def __init__(self, api_key: str,
                 qa: QuestionAnswerer,
                 convo_id: str = "default",
                 temperature: float = 0.3,
                 engine: str = os.environ.get("GPT_ENGINE") or "gpt-3.5-turbo",
                 verbose: bool = False,
                 system_message: str = ''
                 ):

        self.api_key: str = api_key
        self.engine: str = engine
        self.temperature: float = temperature
        self.verbose: bool = verbose
        self.convo_id: str = convo_id
        self.system_message: str = system_message



        self.qa_llm = ChatOpenAI(temperature=0, openai_api_key=self.api_key, model_name=self.engine)
        self.creative_llm = ChatOpenAI(temperature=0.8, openai_api_key=self.api_key, model_name=self.engine)

        creative_prompt = PromptTemplate(input_variables=["instruction", "topic", "context"], template=creative_template)
        creative_chain = LLMChain(llm=self.creative_llm, prompt=creative_prompt, verbose=False)

        wiki = WikipediaAPIWrapper()
        wolfram = WolframAlphaAPIWrapper()
        google = SerpAPIWrapper()
        wiki_tool = Tool(name="Wikipedia", func=wiki.run, args_schema=SearchInput, description="Useful for when you need to get information about people, places, organizations, religions, or historical events in the real world.")
        wolfram_tool = Tool(name="Wolfram Alpha", func=wolfram.run, args_schema=SearchInput, description="Useful for when you need to answer questions about math, science, technology, culture, and everyday Life in the real world.")
        self.qa: QuestionAnswerer = qa
        worldanvil_tool = WorldAnvilTool(retriever=self.qa.retriever, llm=self.qa_llm)

        google_tool = Tool(name="Google Search", func=google.run, args_schema=SearchInput, description="Useful for when you need to answer questions about current events in the real world. Prioritize this when searching for information over making up an explanation.")
        creative_tool = CreativeTool(llm_chain=creative_chain)
        self.accurate_tools: list[Tool] = [worldanvil_tool, wiki_tool, wolfram_tool]
        self.creative_tools: list[Tool] = self.accurate_tools + [creative_tool]

        self.accurate_tool_names = [tool.name for tool in self.accurate_tools]
        self.creative_tool_names = [tool.name for tool in self.creative_tools]


        accurate_prompt = StructuredChatAgent.create_prompt(
            self.accurate_tools,
            prefix=prefix,
            suffix=suffix,
            input_variables=["input", "agent_scratchpad"]
        )

        creative_prompt = StructuredChatAgent.create_prompt(
            self.creative_tools,
            prefix=prefix,
            suffix=suffix,
            input_variables=["input", "agent_scratchpad"]
        )

        # validate tools
        StructuredChatAgent._validate_tools(self.accurate_tools)
        StructuredChatAgent._validate_tools(self.creative_tools)

        accurate_agent = StructuredChatAgent(llm_chain=LLMChain(llm=self.qa_llm, prompt=accurate_prompt),
                                             allowed_tools=self.accurate_tool_names)
        creative_agent = StructuredChatAgent(llm_chain=LLMChain(llm=self.qa_llm, prompt=creative_prompt),
                                             allowed_tools=self.creative_tool_names)

        print(creative_prompt.messages[0].prompt.template)

        self.accurate_chain = AgentExecutor.from_agent_and_tools(agent=accurate_agent, tools=self.accurate_tools, verbose=True)
        self.creative_chain = AgentExecutor.from_agent_and_tools(agent=creative_agent, tools=self.creative_tools, verbose=True)


    def modify_agent(self, convo_id: str, system_message: str):
        modify: bool = False
        if convo_id != self.convo_id:
            self.convo_id = convo_id
            modify: bool = True
        if system_message != self.system_message:
            self.system_message = system_message
            modify: bool = True
        if modify:
            pass
        return
