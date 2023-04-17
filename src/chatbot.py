from typing import Union
import os
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
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


class LangChainChatbot:
    def __init__(self, api_key: str, system_prompt: str,
                 engine: str = os.environ.get("GPT_ENGINE") or "gpt-3.5-turbo",
                 truncate_limit: int = None,
                 ):
        self.api_key: str = api_key
        self.engine: str = engine
        self.num_system: int = 1 # number of system messages
        self.truncate_limit: int = truncate_limit or (
            30500 if engine == "gpt-4-32k" else 6500 if engine == "gpt-4" else 3500
        )

        self.system_prompt: SystemMessage = SystemMessage(content=system_prompt)
        self.chat: ChatOpenAI = ChatOpenAI(openai_api_key=api_key,
                                           callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

        self.conversation: dict[str, list[Union[SystemMessage, HumanMessage, AIMessage]]] = {
            "default": [self.system_prompt]
        }

    def add_to_conversation(self, prompt: str, role: str, convo_id: str = "default") -> None:
        if role.lower() == "user":
            message = HumanMessage(content=prompt)
        elif role.lower() == 'assistant':
            message = AIMessage(content=prompt)
        self.conversation[convo_id].append(message)
        return

    # https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    def get_token_count(self, convo_id: str = "default") -> int:
        """
        Get token count
        """
        num_tokens = self.chat.get_num_tokens_from_messages(self.conversation[convo_id])
        return num_tokens

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

    def ask(self, prompt: str, role: str = "user", convo_id: str = "default"):
        """
        Human as a question.
        :param prompt: Prompt that the human provides.
        :param role: Role - typically "user", or "ai"
        :param convo_id: conversation ID; a string.
        :return:
        """
        message = HumanMessage(content=prompt)
        # add to convo
        self.add_to_conversation(prompt, role, convo_id)
        # truncate conversation
        self.__truncate_conversation(convo_id=convo_id)
        response = self.chat(self.conversation[convo_id])
        self.add_to_conversation(response.content, "assistant", convo_id)
        return response.content

    def rollback(self, n: int = 1, convo_id: str = "default") -> None:
        """
        Rollback the conversation
        """
        for _ in range(n):
            self.conversation[convo_id].pop()

    def reset(self, convo_id: str = "default", system_prompt: str = None) -> None:
        """
        Reset the conversation
        """
        """
        Reset the conversation
        """
        self.conversation[convo_id] = [
            SystemMessage(content=system_prompt or self.system_prompt.content)
        ]
