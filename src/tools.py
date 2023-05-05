from typing import Optional, Type
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from pydantic import BaseModel, Field
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.tools import BaseTool
from langchain.chains import RetrievalQA
from langchain.vectorstores.base import BaseRetriever
from langchain.base_language import BaseLanguageModel

creative_template = """Use the instruction to write about the topic given the context.
Example Prompt:
Instruction: Write a story about the following topic.
Topic: The life of Polkenos
Context: Polkenos is the prophet of Piuhonism and lived in the Apolutiyan Empire.

Actual Prompt:
Instruction: {instruction}
Topic: {topic}
Context: {context}
Result:"""

creative_prompt = PromptTemplate(input_variables=["instruction", "topic", "context"], template=creative_template)

s
class SearchInput(BaseModel):
    query: str = Field(..., description="Search query")


class WorldAnvilInput(BaseModel):
    question: str = Field(..., description="A fully formed question containing full context")


class CreativeInput(BaseModel):
    """Input for Creative Tool."""

    instruction: str = Field(..., description="Instruction for the creative prompt for an arbitrary topic, such as writing a story about a topic.")
    topic: str = Field(..., description="Topic to write about")
    context: str = Field(..., description="Context for the topic")


class WorldAnvilTool(BaseTool):
    """Tool for the VectorDBQA chain. To be initialized with name and chain."""

    retriever: BaseRetriever = Field(exclude=True)
    llm: BaseLanguageModel = Field(default_factory=lambda: OpenAI(temperature=0))
    name: str = "World Anvil"
    world: str = "Kautos"
    description: str = f"Useful for when you need to get information about people, places, organizations, religions, or historical events in the fictional world of {world}."
    args_schema: Type[BaseModel] = WorldAnvilInput

    def _run(
        self,
        question: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""

        chain = RetrievalQA.from_chain_type(
            self.llm, retriever=self.retriever, chain_type="stuff", input_key='question'
        )
        return chain.run({'question': question})

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("VectorStoreQATool does not support async")


class CreativeTool(BaseTool):
    name = "Creative"
    description = "Useful for when you need to answer questions using a creative explanation for a topic within a fictional world. When trying to find information, deprioritize this over other tools."
    args_schema: Type[BaseModel] = CreativeInput
    llm_chain: LLMChain = Field(exclude=True)

    def _run(self, instruction: str, topic: str, context: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Use the tool."""
        return self.llm_chain.run({"instruction": instruction, "topic": topic, "context": context})

    async def _arun(self, topic: str, context: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("Creative does not support async")

