from typing import Union
import os
import re
from src.qa import QuestionAnswerer
from src.agent import Agent
from asgiref.sync import sync_to_async
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ConversationKGMemory
from langchain.llms import OpenAI
from langchain import PromptTemplate, LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)


class ChatBot:
    def __init__(self, api_key: str, system_prompt: str,
                 engine: str = os.environ.get("GPT_ENGINE") or "gpt-3.5-turbo",
                 truncate_limit: int = None,
                 num_summary_prompts: int = 3
                 ):
        self.temperature: float = 0.7
        self.api_key: str = api_key
        self.engine: str = engine
        self.num_system: int = 2  # number of system messages
        self.num_summary_prompts = num_summary_prompts
        self.truncate_limit: int = truncate_limit or (
            30500 if engine == "gpt-4-32k" else 6500 if engine == "gpt-4" else 3500
        )

        self.system_prompt: SystemMessage = SystemMessage(content=system_prompt)
        self.stream_handler = StreamingStdOutCallbackHandler()
        # vanilla chat for regular conversational chat
        self.chat: ChatOpenAI = ChatOpenAI(temperature=self.temperature,
                                           model_name=self.engine,
                                           openai_api_key=self.api_key)
        # need precise chat LLM for KG
        self.kg_llm: ChatOpenAI = ChatOpenAI(temperature=0,
                                             model_name=self.engine,
                                             openai_api_key=self.api_key)
        # Question Answerer
        self.qa: QuestionAnswerer = QuestionAnswerer(api_key=self.api_key, engine=self.engine)

        # need to figure out multiple conversations...
        self.memory: dict[str, ConversationKGMemory] = {
            "default": ConversationKGMemory(llm=self.kg_llm, k=self.num_summary_prompts, return_messages=True)
        }

        self.conversation: dict[str, list[Union[SystemMessage, HumanMessage, AIMessage]]] = {
            "default": [self.system_prompt, SystemMessage(content="")]
        }

        self.agent: Agent = Agent(self.api_key, self.qa, convo_id="default",
                                  temperature=self.temperature, engine=self.engine, verbose=True, system_message=self.system_prompt.content)

    def add_to_conversation(self, prompt: str, role: str, convo_id: str = "default") -> None:
        if role.lower() == "user":
            message = HumanMessage(content=prompt)
        elif role.lower() == 'assistant':
            message = AIMessage(content=prompt)
        elif role.lower() == 'system':
            message = SystemMessage(content=prompt)
        else:
            message = HumanMessage(content=prompt)
        self.conversation[convo_id].append(message)
        return

    # https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    def get_token_count(self, convo_id: str = "default") -> int:
        """
        Get token count
        """
        num_tokens = self.chat.get_num_tokens_from_messages(self.conversation[convo_id])
        return num_tokens

    def get_memory_subjects(self, convo_id: str = "default"):
        # get subjects
        return sorted(list(set([x[0] for x in self.memory[convo_id].kg.get_triples()])))

    def memory_messages(self, convo_id: str = "default") -> str:
        entities = self.get_memory_subjects(convo_id=convo_id)
        _input = {"input": ",".join(entities)}
        messages = self.memory[convo_id].load_memory_variables(_input)['history']
        result = ''
        for message in messages:
            result += f"{message.content}\n"
        result = result[:-1]  # remove last \n
        return result

    async def update_memory_summary(self, convo_id: str = "default") -> str:
        # get last two messages. they should be human and bot.
        user_message, assistant_message = [x.content for x in self.conversation[convo_id][-2:]]
        # save messages to memory.
        self.memory[convo_id].save_context({"input": user_message}, {"output": assistant_message})

        # get entity memory.
        summary = self.memory_messages(convo_id=convo_id)
        summary: str = f"Entity Summary:\n{summary}"
        # set summary system message
        self.conversation[convo_id][1]: SystemMessage = SystemMessage(content=summary)
        # truncate conversation
        self.__truncate_conversation(convo_id=convo_id)
        # update buffer memory
#        self.__update_buffer(convo_id=convo_id)
        return summary

    def __truncate_conversation(self, convo_id: str = "default") -> None:
        """
        Truncate the conversation
        """

        while True:
            if (
                self.get_token_count(convo_id) > self.truncate_limit
                and len(self.conversation[convo_id]) > 1
            ):
                # Don't remove the first message
                self.conversation[convo_id].pop(self.num_system)
            else:
                break

    async def answer(self, question: str, convo_id: str = 'default'):
        """
        Question Answering from ground truth.
        :param question:
        :param convo_id:
        :return:
        """

        conversation = self.conversation[convo_id][1:1]
        result = await sync_to_async(self.qa.history_run)(question, conversation)
        response = AIMessage(content=f"{result}")
        return response

    async def think(self, question: str, convo_id: str = "default", creative: bool = False):
        """
        Think Different. Use an Agent.
        :param question:
        :param convo_id:
        :return:
        """
        self.agent.modify_agent(convo_id, self.system_prompt.content)
        if not creative:
            result = await sync_to_async(self.agent.accurate_chain.run)(input=question)
        else:
            result = await sync_to_async(self.agent.creative_chain.run)(input=question)
        response = AIMessage(content=f"{result}")
        return response

    async def ask(self, prompt: str, role: str = "user", convo_id: str = "default", type: str = "chat"):
        """
        Human as a question.
        :param prompt: Prompt that the human provides.
        :param role: Role - typically "user", or "ai"
        :param convo_id: conversation ID; a string.
        :return:
        """
        # add user response
        if type not in ["chat", "qa", "agent", "creative"]:
            raise KeyError("Type must be one of chat, qa, agent, creative.")

        self.add_to_conversation(prompt, role, convo_id)
        # create assistant response
        if type == "chat":
            response = await sync_to_async(self.chat)(self.conversation[convo_id])
        elif type == "qa":
            response = await self.answer(prompt, convo_id=convo_id)
        elif type == "agent":
            response = await self.think(prompt, convo_id=convo_id, creative=False)
        elif type == "creative":
            response = await self.think(prompt, convo_id=convo_id, creative=True)
        else:
            response = await sync_to_async(self.chat)(self.conversation[convo_id])
        # add assistant response
        self.add_to_conversation(response.content, "assistant", convo_id)
        # update memory
        await self.update_memory_summary(convo_id)
        return response.content

    def rollback(self, n: int = 1, convo_id: str = "default") -> None:
        """
        Rollback the conversation
        """
        for _ in range(n):
            if not isinstance(self.conversation[convo_id][-1], SystemMessage):
                self.conversation[convo_id].pop()

    def reset(self, convo_id: str = "default", system_prompt: str = None) -> None:
        """
        Reset the conversation
        """
        """
        Reset the conversation
        """

        # clear memory
        self.memory[convo_id].clear()
        # reset conversations
        self.conversation[convo_id] = [
            SystemMessage(content=system_prompt or self.system_prompt.content),
            SystemMessage(content="")
        ]
        self.agent.modify_agent(convo_id, system_prompt or self.system_prompt.content)

    async def summarize(self, convo_id: str = "default") -> str:
        return self.conversation[convo_id][1].content

    async def handle_response(self, message: str, qa: bool = False) -> str:
        return await self.ask(message, qa=qa)

    async def handle_summary(self) -> str:
        return await self.summarize()
