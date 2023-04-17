import os
from typing import Optional
from langchain.callbacks.base import AsyncCallbackManager
from langchain.callbacks.tracers import LangChainTracer
from langchain.chains import ChatVectorDBChain
from langchain.chains.chat_vector_db.prompts import (CONDENSE_QUESTION_PROMPT,
                                                     QA_PROMPT)
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.vectorstores.base import VectorStore
from langchain.indexes import VectorstoreIndexCreator
from src.loader import WorldAnvilLoader


class QuestionAnswerer:
    # vanilla LLM for QA
    def __init__(self, api_key: str,
                 engine: str = os.environ.get("GPT_ENGINE") or "gpt-3.5-turbo",
                 tracing: bool = False
                 ):
        # LLM
        self.api_key: str = api_key
        self.engine: str = engine

        # loader
        self.loader = WorldAnvilLoader()

        # figure this out!
        self.tracing: bool = tracing
        self.vector_store: Optional[VectorStore] = self.loader.load_and_create_index()
        self.question_handler = AsyncCallbackManager([])
        self.stream_handler = AsyncCallbackManager([])
        # LLM chain
        self.chain = self.get_chain(tracing=tracing)

    def get_chain(self, tracing: bool = False
    ) -> ChatVectorDBChain:
        """Create a ChatVectorDBChain for question/answering."""
        # Construct a ChatVectorDBChain with a streaming llm for combine docs
        # and a separate, non-streaming llm for question generation
        manager = AsyncCallbackManager([])
        if tracing:
            tracer = LangChainTracer()
            tracer.load_default_session()
            manager.add_handler(tracer)
            self.question_handler.add_handler(tracer)
            self.stream_handler.add_handler(tracer)

        question_gen_llm = OpenAI(
            temperature=0,
            verbose=True,
            callback_manager=self.question_handler,
            open_api_key=self.api_key,
            engine=self.engine
        )
        streaming_llm = OpenAI(
            streaming=True,
            temperature=0,
            verbose=True,
            callback_manager=self.stream_handler,
            open_api_key=self.api_key,
            engine=self.engine
        )

        question_generator = LLMChain(
            llm=question_gen_llm, prompt=CONDENSE_QUESTION_PROMPT, callback_manager=manager
        )
        doc_chain = load_qa_chain(
            streaming_llm, chain_type="stuff", prompt=QA_PROMPT, callback_manager=manager
        )

        qa = ChatVectorDBChain(
            vectorstore=self.vector_store,
            combine_docs_chain=doc_chain,
            question_generator=question_generator,
            callback_manager=manager,
        )
        return qa
