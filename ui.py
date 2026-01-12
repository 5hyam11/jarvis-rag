import streamlit as st
import requests

API_URL = "https://jarvis-api.onrender.com"


st.set_page_config(
    page_title="Jarvis",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Jarvis")
st.caption("Memory-backed AI assistant")

# -------------------------
# Sidebar: File upload
# -------------------------
st.sidebar.header("📂 Knowledge Base")

uploaded_file = st.sidebar.file_uploader(
    "Upload a document",
    type=["pdf", "txt", "md"],
)

if uploaded_file:
    with st.sidebar.status("Ingesting document...", expanded=False):
        response = requests.post(
            f"{API_URL}/ingest",
            files={"file": (uploaded_file.name, uploaded_file.getvalue())},
        )

        if response.ok:
            data = response.json()
            st.sidebar.success(
                f"Ingested {data['chunks_added']} chunks from {data['filename']}"
            )
        else:
            st.sidebar.error("Failed to ingest file")

# -------------------------
# Chat state
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------------
# Chat input
# -------------------------
prompt = st.chat_input("Ask Jarvis something...")

if prompt:
    # Show user message
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )
    with st.chat_message("user"):
        st.markdown(prompt)

    # Ask backend
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = requests.post(
                f"{API_URL}/chat",
                params={"question": prompt},
            )

            if response.ok:
                result = response.json()
                answer = result["answer"]
                citations = result["citations"]

                st.markdown(answer)

                if citations:
                    with st.expander("Sources"):
                        for i, c in enumerate(citations, 1):
                            st.markdown(
                                f"**{i}. {c['source']}**  \n"
                                f"Score: `{c['score']}`  \n"
                                f"> {c['text']}"
                            )

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )
            else:
                st.error("Jarvis failed to respond")
