from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import Settings


class IngestionResult:
    def __init__(self, vectorstore: FAISS, chunk_count: int, source_names: list[str]):
        self.vectorstore = vectorstore
        self.chunk_count = chunk_count
        self.source_names = source_names


def build_vectorstore(documents: list[Document], settings: Settings) -> IngestionResult:
    if not documents:
        raise ValueError("No documents provided for ingestion.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("Document splitting produced no chunks.")

    embeddings = OpenAIEmbeddings(model=settings.embedding_model)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    source_names = sorted({doc.metadata.get("source", "unknown") for doc in documents})

    return IngestionResult(
        vectorstore=vectorstore,
        chunk_count=len(chunks),
        source_names=source_names,
    )


def build_retriever(vectorstore: FAISS, settings: Settings) -> VectorStoreRetriever:
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": settings.retrieval_k},
    )
