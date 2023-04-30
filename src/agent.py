import os
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.utilities import WikipediaAPIWrapper, WolframAlphaAPIWrapper, SerpAPIWrapper
from src.qa import QuestionAnswerer


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

        template = """Write a creative explanation for the following topic.
        Topic: {input}
        Explanation:"""

        self.qa_llm = OpenAI(temperature=0, openai_api_key=self.api_key, model_name=self.engine)
        self.creative_llm = OpenAI(temperature=0.8, openai_api_key=self.api_key, model_name=self.engine)

        prompt = PromptTemplate(input_variables=["input"], template=template)

        creative_chain = LLMChain(llm=self.creative_llm, prompt=prompt, verbose=False)

        wiki = WikipediaAPIWrapper()
        wolfram = WolframAlphaAPIWrapper()
        google = SerpAPIWrapper()
        wiki_tool = Tool(name="Wikipedia", func=wiki.run, description="Useful for when you need to get information about people, places, organizations, religions, or historical events on Earth, or drawing inspiration from these things on Earth as a creative input for Kautos. When comparing things from Earth and Kautos, search this for things from Earth, and use World Anvil for things on Kautos. Input should be a search query.")
        wolfram_tool = Tool(name="Wolfram Alpha", func=wolfram.run, description="Useful for when you need to answer questions about Math, Science, Technology, Earth Culture, and Everyday Life. Prioritize this when searching for information over making up an explanation. Input should be a search query.")
        self.qa: QuestionAnswerer = qa
        worldanvil_tool = Tool(name="World Anvil", func=self.qa.chain_run, description="Useful for when you need to get information about people, places, organizations, religions, or historical events on Kautos. Search this when information cannot be found on Wikipedia. When comparing things from Earth and Kautos, search this for things from Kautos, and use Wikipedia for things on Earth. Input should be a search query.")
        google_tool = Tool(name="Google Search", func=google.run, description="Useful for when you need to answer questions about current events, or when information cannot be found in Wikipedia or World Anvil. Prioritize this when searching for information over making up an explanation. Input should be a search query.")
        creative_tool = Tool(name="Creative", func=creative_chain.run, description="Useful for when you need to answer questions using a creative explanation for a topic within Kautos. Deprioritize this over Wikipedia and World Anvil. Input should be a search query.")
        self.accurate_tools: list[Tool] = [worldanvil_tool, wiki_tool, google_tool]
        self.creative_tools: list[Tool] = self.accurate_tools + [creative_tool]

        self.accurate_chain = initialize_agent(self.accurate_tools, self.qa_llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                                               verbose=self.verbose)
        self.creative_chain = initialize_agent(self.creative_tools, self.qa_llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                                      verbose=self.verbose)


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
