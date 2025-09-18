import streamlit as st
import os
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from typing import List
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate
)
import pdfplumber
import requests
import re


st.markdown("""
<style>
    /* Existing styles */
    .main {
        background-color: #1a1a1a;
        color: #ffffff;
    }
    .sidebar .sidebar-content {
        background-color: #2d2d2d;
    }
    .stTextInput textarea {
        color: #ffffff !important;
    }
    
    /* Add these new styles for select box */
    .stSelectbox div[data-baseweb="select"] {
        color: white !important;
        background-color: #3d3d3d !important;
    }
    
    .stSelectbox svg {
        fill: white !important;
    }
    
    .stSelectbox option {
        background-color: #2d2d2d !important;
        color: white !important;
    }
    
    /* For dropdown menu items */
    div[role="listbox"] div {
        background-color: #2d2d2d !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)



if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []


system_prompt = SystemMessagePromptTemplate.from_template(
   """You are an expert research assistant. Use the provided context to answer the query. 
If unsure, state that you don't know."""

)


#UI Elements
st.title("AI Assistant")
st.markdown("### Your Intelligent Research Assistant")
st.markdown("---")

with st.sidebar:
    st.header("⚙️ Configuration")
    selected_model = st.selectbox(
        "Choose Model",
        ["deepseek-r1:8b", "qwen3:8b"],
        index=0
    )
    st.divider()
    st.markdown("### Model Capabilities")
    st.markdown("""
    - PDF assistant
    - Helps summarize PDFs
            
    """)
PDF_STORAGE_PATH = 'document_store/'
OLLAMA_SERVER_URL = "http://127.0.0.1:8080"
MAX_PDF_PAGES = 20  # Set your desired page limit
uploaded = False

os.makedirs(PDF_STORAGE_PATH, exist_ok=True)

class RemoteOllamaEmbeddings(Embeddings):
    def __init__(self, model_name: str, server_url: str):
        self.model_name = model_name
        self.server_url = server_url
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = requests.post(
            f"{self.server_url}/embed",
            json={"texts": texts, "model_name": self.model_name}
        )
        response.raise_for_status()
        return response.json().get("embeddings", [])
    
    def embed_query(self, text: str) -> List[float]:
        response = requests.post(
            f"{self.server_url}/embed",
            json={"texts": [text], "model_name": self.model_name}
        )
        response.raise_for_status()
        embeddings = response.json().get("embeddings", [])
        return embeddings[0] if embeddings else []

# Then replace your vector store initialization with:
DOCUMENT_VECTOR_DB = InMemoryVectorStore(embedding=RemoteOllamaEmbeddings(
    model_name=selected_model, 
    server_url=OLLAMA_SERVER_URL
))

if 'vector_store' not in st.session_state:
    st.session_state.vector_store = InMemoryVectorStore(embedding=RemoteOllamaEmbeddings(
        model_name=selected_model,
        server_url=OLLAMA_SERVER_URL
    ))



def save_uploaded_file(uploaded_file):
    file_path = PDF_STORAGE_PATH + uploaded_file.name
    with open(file_path,"wb") as file:
        file.write(uploaded_file.getbuffer())
    return file_path



def load_pdf_documents(file_path):
    document_loader = PDFPlumberLoader(file_path)
    return document_loader.load()

def chunk_documents(raw_documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        add_start_index = True
    )
    return text_splitter.split_documents(raw_documents)

def index_documents(document_chunks):
    """
    Adds document chunks to the vector store using add_texts, which will call the embedding function automatically.
    """
    try:
        texts = [doc.page_content for doc in document_chunks]
        metadatas = [doc.metadata for doc in document_chunks]
        DOCUMENT_VECTOR_DB.add_texts(texts, metadatas)
    except Exception as e:
        st.error(f"Error adding documents to the vector store: {e}")

def find_related_documents(query):
    """
    Finds related documents using similarity search in the vector store.
    """
    return DOCUMENT_VECTOR_DB.similarity_search(query)

def build_prompt_chain(user_query, context_documents):
    context = "\n\n".join([doc.page_content for doc in context_documents])
    prompt_sequence = [system_prompt]
    for msg in st.session_state.message_log:
        if msg["role"] == "user":
            prompt_sequence.append(HumanMessagePromptTemplate.from_template(msg["content"]))
        elif msg["role"] == "ai":
            prompt_sequence.append(AIMessagePromptTemplate.from_template(msg["content"]))
    
    prompt_sequence.append(
        HumanMessagePromptTemplate.from_template(
            f"Context: {context}\n\nQuery: {user_query}"
        )
    )
    return ChatPromptTemplate.from_messages(prompt_sequence)

def generate_ai_response(prompt_chain, selected_model):
    """
    Sends the prompt and selected model to the Ollama server and retrieves the response.
    """
    prompt = str(prompt_chain)
    response = requests.post(
        f"{OLLAMA_SERVER_URL}/ollama",
        json={"query": prompt, "model_name": selected_model}
    )
    try:
        return response.json().get("response", "")
    except Exception as e:
        st.error(
            f"Server did not return valid JSON. "
            f"Status code: {response.status_code}, Content: {response.text}"
        )
        return ""

   


uploaded_pdf = st.file_uploader("Upload Research Document (PDF)",
                                type = "pdf",
                                help="Select a PDF document to analysis",
                                accept_multiple_files=False)

if uploaded_pdf:
    # Save the file temporarily to check pages
    saved_path = save_uploaded_file(uploaded_pdf)
    with pdfplumber.open(saved_path) as pdf:
        num_pages = len(pdf.pages)
    if num_pages > MAX_PDF_PAGES:
        st.error(f"❌ PDF has {num_pages} pages. Please upload a PDF with {MAX_PDF_PAGES} pages or fewer.")
        os.remove(saved_path)  # Optionally remove the file if too large
    else:
        raw_docs = load_pdf_documents(saved_path)
        processed_chunks = chunk_documents(raw_docs)
        index_documents(processed_chunks)
        uploaded = True
        st.success("✅ Document processed successfully! Ask your questions below.")


if "message_log" not in st.session_state:
    st.session_state.message_log = [{"role": "ai", "content": "Hi! I'm your AI assistant. How can I help you today?"}]
def render_response_with_math_and_thinking(content):
    """
    Renders LLM response, displaying <think>...</think> text in a lighter color,
    and LaTeX math expressions as equations.
    """

    # Split content into segments: <think>, LaTeX, and normal text
    pattern = re.compile(
        r"(<think>.*?</think>)|"
        r"(\\\((.*?)\\\))|"
        r"(\\\[(.*?)\\\])",
        re.DOTALL
    )

    pos = 0
    for match in pattern.finditer(content):
        start, end = match.span()
        # Render any normal text before this match
        if start > pos:
            st.markdown(content[pos:start], unsafe_allow_html=True)
        if match.group(1):  # <think>...</think>
            inner = match.group(1)[7:-8]  # Remove <think> and </think>
            st.markdown(
                f'<span style="color:#bbbbbb;font-style:italic;">{inner}</span>',
                unsafe_allow_html=True
            )
        elif match.group(2):  # \( ... \)
            st.latex(match.group(3).strip())
        elif match.group(4):  # \[ ... \]
            st.latex(match.group(5).strip())
        pos = end
    # Render any remaining text after the last match
    if pos < len(content):
        st.markdown(content[pos:], unsafe_allow_html=True)

# Chat container to display messages
chat_container = st.container()
with chat_container:
    for message in st.session_state.message_log:
        with st.chat_message(message["role"]):
            if message["role"] == "ai":
                render_response_with_math_and_thinking(message["content"])  # Render AI messages with math support
            else:
                st.markdown(message["content"])  # Render user messages as plain text

# Chat input for user queries
user_input = st.chat_input("Enter your question about the document...")

if user_input:
    # Check if any documents have been indexed
    if not uploaded:
       st.error("❌ Please upload and process a PDF before asking questions.")
    else:
    # ...rest of your code...
        # Append user input to the message log
        st.session_state.message_log.append({"role": "user", "content": user_input})

        with st.spinner("🧠 Thinking..."):
            relevant_docs = find_related_documents(user_input)
            prompt_chain = build_prompt_chain(user_input, relevant_docs)
            ai_response = generate_ai_response(prompt_chain, selected_model)  # Pass the selected model

        # Append AI response to the message log
        st.session_state.message_log.append({"role": "ai", "content": ai_response})

        # Rerun the app to update the chat UI
        st.rerun()
