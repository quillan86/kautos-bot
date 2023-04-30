import os
import re
from typing import Optional
from langchain.callbacks.base import AsyncCallbackManager
from langchain.callbacks.tracers import LangChainTracer
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chains.chat_vector_db.prompts import (CONDENSE_QUESTION_PROMPT,
                                                     QA_PROMPT)
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.vectorstores.base import VectorStore
from src.loader import WorldAnvilLoader


class QuestionAnswerer:
    # vanilla LLM for QA
    def __init__(self, api_key: str,
                 engine: str = os.environ.get("GPT_ENGINE") or "gpt-3.5-turbo",
                 tracing: bool = False
                 ):
        # LLM
        self.temperature: float = 0.2  # for some originality
        self.k: int = 4 # k in kNN
        self.api_key: str = api_key
        self.engine: str = engine

        # loader
        self.loader = WorldAnvilLoader()

        # figure this out!
        self.tracing: bool = tracing
        self.vector_store: Optional[VectorStore] = self.loader.load_and_create_index()
        self.retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": self.k})
        self.question_handler = AsyncCallbackManager([])
        self.stream_handler = AsyncCallbackManager([])
        # LLM chain
        self.history_chain = self.get_chain(tracing=tracing)
        self.chain = self.get_retriever(tracing=tracing)

    def get_chain(self, tracing: bool = False
    ) -> ConversationalRetrievalChain:
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
            temperature=0.0,
            model_name=self.engine,
            verbose=True,
            callback_manager=self.question_handler,
            openai_api_key=self.api_key
        )
        streaming_llm = OpenAI(
            streaming=True,
            temperature=self.temperature,
            model_name=self.engine,
            verbose=True,
            callback_manager=self.stream_handler,
            openai_api_key=self.api_key
        )

        question_generator = LLMChain(
            llm=question_gen_llm, prompt=CONDENSE_QUESTION_PROMPT, callback_manager=manager
        )
        doc_chain = load_qa_chain(
            streaming_llm, chain_type="stuff", prompt=QA_PROMPT, callback_manager=manager
        )

        qa = ConversationalRetrievalChain(
            retriever=self.retriever,
            combine_docs_chain=doc_chain,
            question_generator=question_generator,
            callback_manager=manager,
            return_source_documents=True
        )
        return qa

    def get_retriever(self, tracing: bool = False
    ) -> RetrievalQA:
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

        streaming_llm = OpenAI(
            streaming=True,
            temperature=self.temperature,
            model_name=self.engine,
            verbose=True,
            callback_manager=self.stream_handler,
            openai_api_key=self.api_key
        )

        doc_chain = load_qa_chain(
            streaming_llm, chain_type="stuff", prompt=QA_PROMPT, callback_manager=manager
        )

        qa = RetrievalQA(
            retriever=self.retriever,
            combine_documents_chain=doc_chain,
            callback_manager=manager,
            return_source_documents=True
        )
        return qa

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

    def chain_run(self, question):
        result = self.chain({"query": question})
        answer = result["result"]
        sources = result["source_documents"]

        if len(sources) > 0:
            citations = self.get_citations(sources)
            response = f"{answer}\nSources: {citations}"
        else:
            response = f"{answer}"
        return response

    def history_run(self, question, conversation) -> str:
        chat_history = [x.content for x in conversation]

        # get answer
        result = self.history_chain(
            {"question": question, "chat_history": chat_history}
        )
        answer = result["answer"]
        sources = result["source_documents"]

        if len(sources) > 0:
            citations = self.get_citations(sources)
            response = f"{answer}\nSources: {citations}"
        else:
            response = f"{answer}"
        return response
