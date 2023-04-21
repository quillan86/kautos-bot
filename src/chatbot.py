from typing import Union
import os
import re
from src.qa import QuestionAnswerer
from asgiref.sync import sync_to_async
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import CallbackManager
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
#                                           engine=self.engine,
                                           openai_api_key=self.api_key,
                                           callback_manager=CallbackManager([self.stream_handler]))
        # need precise chat LLM for KG
        self.memory_llm: ChatOpenAI = ChatOpenAI(temperature=0,
#                                                 engine=self.engine,
                                                 openai_api_key=self.api_key,
                                                 callback_manager=CallbackManager([self.stream_handler]))
        # Question Answerer
        self.qa: QuestionAnswerer = QuestionAnswerer(api_key=self.api_key, engine=self.engine)
        # need to figure out multiple conversations...
        self.memory: dict[str, ConversationKGMemory] = {
            "default": ConversationKGMemory(llm=self.memory_llm, k=self.num_summary_prompts, return_messages=True)
        }

        self.conversation: dict[str, list[Union[SystemMessage, HumanMessage, AIMessage]]] = {
            "default": [self.system_prompt, SystemMessage(content="")]
        }

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

    def fix_citation_label(self, name: str):
        pattern = r'[^a-zA-Z\s]*'  # Matches any non-alphanumeric characters and digits
        result = re.sub(pattern, '', name).strip()
        return result

    def get_citations(self, documents) -> str:
        """
        [citation needed]
        :param documents: source documents
        :return: [citations]!!
        """
        # unique citations
        citations = set([f"[{self.fix_citation_label(x.metadata['name'])}]({x.metadata['url']})" for x in documents])
        # convert to string
        result = ', '.join(citations)
        return result

    async def answer(self, question: str, convo_id: str = 'default'):
        """
        Question Answering from ground truth.
        :param question:
        :param convo_id:
        :return:
        """
        chat_history = [x.content for x in self.conversation[convo_id][1:1]]

        # get answer
        result = self.qa.chain(
            {"question": question, "chat_history": chat_history}
        )
        answer = result["answer"]
        sources = result["source_documents"]

        if len(sources) > 0:
            citations = self.get_citations(sources)
            response = AIMessage(content=f"{answer}\nSources: {citations}")
        else:
            response = AIMessage(content=f"{answer}")
        return response

    async def ask(self, prompt: str, role: str = "user", convo_id: str = "default", qa: bool = False):
        """
        Human as a question.
        :param prompt: Prompt that the human provides.
        :param role: Role - typically "user", or "ai"
        :param convo_id: conversation ID; a string.
        :return:
        """
        # add user response
        self.add_to_conversation(prompt, role, convo_id)
        # create assistant response
        if qa is False:
            response = await sync_to_async(self.chat)(self.conversation[convo_id])
        else:
            response = await self.answer(prompt, convo_id=convo_id)
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

    async def summarize(self, convo_id: str = "default") -> str:
        return self.conversation[convo_id][1].content

    async def handle_response(self, message: str, qa: bool = False) -> str:
        return await self.ask(message, qa=qa)

    async def handle_summary(self) -> str:
        return await self.summarize()
