"""Optional Streamlit UI.  Run:  streamlit run streamlit_app.py"""
import streamlit as st

from src import config
from src.graph import ask, build_rag_graph

st.set_page_config(page_title="Agentic AI RAG Chatbot", layout="wide")


@st.cache_resource
def get_graph():
    return build_rag_graph(index_name=config.PINECONE_INDEX_NAME)


st.title("Agentic AI eBook Chatbot")
st.caption("Answers strictly from the eBook. Off-topic questions are refused.")

if "history" not in st.session_state:
    st.session_state.history = []
if "last" not in st.session_state:
    st.session_state.last = None

for role, msg in st.session_state.history:
    st.chat_message(role).write(msg)

if query := st.chat_input("Ask something about the eBook..."):
    st.chat_message("user").write(query)
    with st.spinner("Thinking..."):
        result = ask(get_graph(), query)
    st.chat_message("assistant").write(result["answer"])
    st.session_state.history += [("user", query), ("assistant", result["answer"])]
    st.session_state.last = result

with st.sidebar:
    st.header("Retrieval details")
    last = st.session_state.last
    if last:
        st.metric("Confidence score", f"{last['score']:.2f}")
        for i, c in enumerate(last["context"], 1):
            with st.expander(f"Chunk {i} - page {c['page']} - sim {c['score']:.3f}"):
                st.write(c["text"])
    else:
        st.info("Ask a question to see retrieved chunks.")
