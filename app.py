import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from rag import load_vectorstore, build_retriever, ask_rag


st.set_page_config(
    page_title="RAG Assistant",
    page_icon="📚",
)

st.title("📚 RAG Assistant")
st.write("Ask questions based on the documents in the knowledge base.")


@st.cache_resource
def load_rag_components():
    vectorstore = load_vectorstore()
    retriever = build_retriever(vectorstore)
    return retriever


retriever = load_rag_components()


with st.sidebar:
    st.header("API key")
    api_key = st.text_input(
        "Enter your Google AI Studio API key.",
        type="password",
        key="google_api_key",
    )

api_key = (api_key or "").strip()

question = st.chat_input("Enter your question", disabled=not api_key)

if question and api_key:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching documents..."):
                answer, sources = ask_rag(question, retriever, api_key)
        except Exception as e:
            msg = str(e)
            if any(s in msg for s in ("API key", "API_KEY", "PERMISSION_DENIED", "401", "403")):
                st.error("Invalid or incorrect API key. Please check the key and try again.")
            else:
                st.error(f"Error: {msg}")
        else:
            st.write(answer)

            if sources:
                st.markdown("**Sources:**")
                for source in sources:
                    st.write(f"- {source}")