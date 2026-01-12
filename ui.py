import os
import streamlit as st
from openai import OpenAI as OpenAIClient

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Jarvis", layout="wide")
st.title("🧠 Jarvis")
st.caption("Memory-backed AI assistant (Streamlit-only)")

# -----------------------------
# Load API key
# -----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not found")
    st.stop()

client = OpenAIClient(api_key=OPENAI_API_KEY)

# -----------------------------
# Initialize LLM + embeddings
# -----------------------------
llm = OpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
embed_model = OpenAIEmbedding(api_key=OPENAI_API_KEY)

# -----------------------------
# Sidebar: Knowledge Base
# -----------------------------
st.sidebar.header("📚 Knowledge Base")

uploaded_files = st.sidebar.file_uploader(
    "Upload documents",
    type=["pdf", "txt", "md"],
    accept_multiple_files=True
)

os.makedirs("data", exist_ok=True)

if uploaded_files:
    for file in uploaded_files:
        with open(f"data/{file.name}", "wb") as f:
            f.write(file.read())
    st.sidebar.success("Documents uploaded")

# -----------------------------
# Build / load index
# -----------------------------
@st.cache_resource(show_spinner=True)
def load_index():
    docs = SimpleDirectoryReader("data").load_data()
    return VectorStoreIndex.from_documents(docs, embed_model=embed_model)

index = load_index() if os.listdir("data") else None

# -----------------------------
# Chat state
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# -----------------------------
# Voice input
# -----------------------------
st.subheader("🎤 Talk to Jarvis")

audio_file = st.file_uploader(
    "Record your voice and upload it",
    type=["wav", "mp3", "m4a"]
)

prompt = None

if audio_file is not None:
    with open("voice_input.wav", "wb") as f:
        f.write(audio_file.read())

    with st.spinner("Listening..."):
        transcript = client.audio.transcriptions.create(
            file=open("voice_input.wav", "rb"),
            model="gpt-4o-transcribe"
        )

    prompt = transcript.text
    st.success(f"You said: {prompt}")

# -----------------------------
# Text input fallback
# -----------------------------
text_prompt = st.chat_input("Ask Jarvis something...")

if text_prompt:
    prompt = text_prompt

# -----------------------------
# Generate response
# -----------------------------
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    try:
        if index:
            query_engine = index.as_query_engine(llm=llm)
            answer = str(query_engine.query(prompt))
        else:
            answer = llm.complete(prompt).text

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )
        st.chat_message("assistant").write(answer)

        # -----------------------------
        # Text → Speech
        # -----------------------------
        speech = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="alloy",
            input=answer
        )

        with open("jarvis_reply.mp3", "wb") as f:
            f.write(speech.read())

        st.audio("jarvis_reply.mp3", format="audio/mp3")

    except Exception as e:
        st.error("Jarvis failed to respond")
        st.exception(e)
