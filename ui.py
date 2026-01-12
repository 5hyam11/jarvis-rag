
import os
import tempfile
import numpy as np
import streamlit as st
import soundfile as sf
import av

from streamlit_webrtc import webrtc_streamer, AudioProcessorBase

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# =========================================================
# Page config
# =========================================================
st.set_page_config(page_title="Jarvis", layout="wide")

st.title("🧠 Jarvis")
st.caption("Live voice-enabled, memory-backed AI assistant (Groq-powered)")

# =========================================================
# Load API key (ONLY Groq)
# =========================================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not found. Add it to Streamlit / Render environment variables.")
    st.stop()

# =========================================================
# Initialize models
# =========================================================
llm = Groq(
    model="llama-3.1-70b-versatile",
    api_key=GROQ_API_KEY,
)


embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)


# =========================================================
# Sidebar: Knowledge Base
# =========================================================
st.sidebar.header("📚 Knowledge Base")

uploaded_files = st.sidebar.file_uploader(
    "Upload documents (PDF, TXT, MD)",
    type=["pdf", "txt", "md"],
    accept_multiple_files=True
)

os.makedirs("data", exist_ok=True)

if uploaded_files:
    for file in uploaded_files:
        with open(f"data/{file.name}", "wb") as f:
            f.write(file.read())
    st.sidebar.success("Documents uploaded")

# =========================================================
# Build / load index (cached)
# =========================================================
@st.cache_resource(show_spinner=True)
def load_index():
    docs = SimpleDirectoryReader("data").load_data()
    return VectorStoreIndex.from_documents(
        docs,
        embed_model=embed_model
    )

index = load_index() if os.listdir("data") else None

# =========================================================
# Session state
# =========================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# =========================================================
# 🎤 Live microphone (capture only, text-based)
# =========================================================
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.frames = []

    def recv(self, frame: av.AudioFrame):
        audio = frame.to_ndarray()
        self.frames.append(audio)
        return frame

st.subheader("🎤 Live Microphone")

webrtc_ctx = webrtc_streamer(
    key="jarvis-mic",
    audio_processor_factory=AudioProcessor,
    media_stream_constraints={"audio": True, "video": False},
)

prompt = None

if webrtc_ctx.audio_processor and st.button("🛑 Stop Recording"):
    frames = webrtc_ctx.audio_processor.frames
    if frames:
        audio_data = np.concatenate(frames, axis=0)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio_data, samplerate=48000)

        st.info("🎧 Audio recorded. (Speech-to-text disabled in free mode)")
        st.warning("Type your message below to continue.")

# =========================================================
# 💬 Text input
# =========================================================
text_prompt = st.chat_input("Ask Jarvis something...")

if text_prompt:
    prompt = text_prompt

# =========================================================
# 🤖 Jarvis response
# =========================================================
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    try:
        if index:
            query_engine = index.as_query_engine(llm=llm)
            response = query_engine.query(prompt)
            answer = str(response)
        else:
            answer = llm.complete(prompt).text

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )
        st.chat_message("assistant").write(answer)

    except Exception as e:
        st.error("Jarvis failed to respond.")
        st.exception(e)
