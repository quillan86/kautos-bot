import os
from typing import Type
import pandas as pd
from langchain.docstore.document import Document
from langchain.vectorstores import OpenSearchVectorSearch, Chroma
from langchain.vectorstores.base import VectorStore
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


class WorldAnvilLoader:
    vectorstore_cls: Type[VectorStore] = Chroma

    def __init__(self, chunk_size: int = 1000):
        self.chunk_size = chunk_size
        self.config_dir = os.path.abspath(f"{__file__}/../../data/")
        self.index_name: str = "world_anvil"

        self.entities_df: pd.DataFrame = pd.read_csv(os.path.join(self.config_dir, "entities.csv"))
        self.kg_df: pd.DataFrame = pd.read_csv(os.path.join(self.config_dir, "kg.csv"))
        self.embedding = OpenAIEmbeddings()
        self.es_url: str = "http://localhost:9200"
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=0)

    def load(self) -> list[Document]:
        docs: list[Document] = []
        for i, row in self.entities_df.iterrows():
            content = row['content']

            metadata = {"source": f"{row['name']} [Doc {i}]",
                        "name": row['name'],
                        "type": row["type"],
                        "url": row["url"]
                        }

            doc = Document(page_content=content, metadata=metadata)
            docs.append(doc)
        return docs

    def create_index(self, docs: list[Document], **kwargs) -> VectorStore:
        sub_docs = self.text_splitter.split_documents(docs)
        ids = [str(x) for x in range(len(sub_docs))]
        vector_store: VectorStore = self.vectorstore_cls.from_documents(
            sub_docs,
            embedding=self.embedding,
            ids=ids,
            index_name=self.index_name,
            **kwargs
        )
        return vector_store

    def load_and_create_index(self, **kwargs) -> VectorStore:
        docs: list[Document] = self.load()
        vector_store: VectorStore = self.create_index(docs, **kwargs)
        return vector_store
