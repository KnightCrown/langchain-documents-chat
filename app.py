import streamlit as st

from config import get_settings
from services.document_loader import DocumentLoadError, load_uploaded_files
from services.ingestion import build_retriever, build_vectorstore
from services.rag_chain import build_conversational_chain, clear_session_history

SESSION_ID = "default"


def init_session_state() -> None:
    defaults = {
        "messages": [],
        "docs_ready": False,
        "rag_chain": None,
        "indexed_files": [],
        "chunk_count": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_chat_state() -> None:
    st.session_state.messages = []
    clear_session_history(SESSION_ID)


def render_sidebar(settings) -> None:
    st.subheader("Document uploader")
    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, or DOCX files",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
    )

    if not settings.has_api_key:
        st.error("Add your OpenAI API key to a `.env` file before processing documents.")

    if st.button("Process documents", disabled=not settings.has_api_key):
        if not uploaded_files:
            st.warning("Select at least one file to process.")
            return

        with st.spinner("Processing documents..."):
            try:
                documents = load_uploaded_files(uploaded_files)
                ingestion = build_vectorstore(documents, settings)
                retriever = build_retriever(ingestion.vectorstore, settings)
                st.session_state.rag_chain = build_conversational_chain(retriever, settings)
                st.session_state.docs_ready = True
                st.session_state.indexed_files = ingestion.source_names
                st.session_state.chunk_count = ingestion.chunk_count
                reset_chat_state()
                st.success(
                    f"Indexed {len(ingestion.source_names)} file(s) "
                    f"into {ingestion.chunk_count} chunks."
                )
            except DocumentLoadError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Failed to process documents: {exc}")

    if st.session_state.docs_ready:
        st.divider()
        st.caption("Indexed files")
        for name in st.session_state.indexed_files:
            st.write(f"- {name}")
        st.caption(f"Chunks: {st.session_state.chunk_count}")
        st.caption(f"Chat model: {settings.chat_model}")
        st.caption(f"Embeddings: {settings.embedding_model}")


def render_chat() -> None:
    if not st.session_state.docs_ready:
        st.info("Upload and process documents in the sidebar before chatting.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input(
        "Ask a question about your documents...",
        disabled=not st.session_state.docs_ready,
    )

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.rag_chain.invoke(
                        {"input": prompt},
                        config={"configurable": {"session_id": SESSION_ID}},
                    )
                    answer = response.get("answer", "I could not generate an answer.")
                except Exception as exc:
                    answer = f"Something went wrong: {exc}"

                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})


def main() -> None:
    settings = get_settings()
    init_session_state()

    st.set_page_config(
        page_title="LangChain Documents Chat",
        page_icon="📄",
        layout="wide",
    )

    st.title("LangChain Documents Chat")
    st.caption("Upload documents, then ask questions grounded in your files.")

    with st.sidebar:
        render_sidebar(settings)

    render_chat()


if __name__ == "__main__":
    main()
