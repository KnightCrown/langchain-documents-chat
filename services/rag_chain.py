from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import ChatOpenAI

from config import Settings

CONTEXTUALIZE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given a chat history and the latest user question, rewrite it as a "
            "standalone question that can be understood without the chat history. "
            "Do not answer the question.",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that answers questions using only the "
            "provided document context. If the answer is not in the context, say "
            "you do not know based on the uploaded documents. Keep answers concise.\n\n"
            "Context:\n{context}",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

_session_histories: dict[str, BaseChatMessageHistory] = {}


def _get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in _session_histories:
        _session_histories[session_id] = ChatMessageHistory()
    return _session_histories[session_id]


def clear_session_history(session_id: str = "default") -> None:
    _session_histories.pop(session_id, None)


def build_conversational_chain(
    retriever: VectorStoreRetriever,
    settings: Settings,
) -> RunnableWithMessageHistory:
    llm = ChatOpenAI(
        model=settings.chat_model,
        temperature=settings.temperature,
        api_key=settings.openai_api_key,
    )

    history_aware_retriever = create_history_aware_retriever(
        llm,
        retriever,
        CONTEXTUALIZE_PROMPT,
    )
    question_answer_chain = create_stuff_documents_chain(llm, ANSWER_PROMPT)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return RunnableWithMessageHistory(
        rag_chain,
        _get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
