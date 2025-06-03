import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate
)
from langchain_ollama.llms import OllamaLLM
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
EMBEDDING_MODEL = OllamaEmbeddings(model=selected_model)
DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)
LANGUAGE_MODEL = OllamaLLM(model=selected_model)

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
    DOCUMENT_VECTOR_DB.add_documents(document_chunks)

def find_related_documents(query):
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

def generate_ai_response(prompt_chain):
    processing_pipeline=prompt_chain | LANGUAGE_MODEL | StrOutputParser()
    return processing_pipeline.invoke({})

   


uploaded_pdf = st.file_uploader("Upload Research Document (PDF)",
                                type = "pdf",
                                help="Select a PDF document to analysis",
                                accept_multiple_files=False)

if uploaded_pdf:
    saved_path = save_uploaded_file(uploaded_pdf)
    raw_docs = load_pdf_documents(saved_path)
    processed_chunks = chunk_documents(raw_docs)
    index_documents(processed_chunks)
    
    st.success("✅ Document processed successfully! Ask your questions below.")


if "message_log" not in st.session_state:
    st.session_state.message_log = [{"role": "ai", "content": "Hi! I'm your AI assistant. How can I help you today?"}]
def render_response_with_math(content):
    """
    Parses the LLM response and renders text and LaTeX math expressions separately.
    """
    import re

    # Regex to find LaTeX math expressions enclosed in \( ... \) or \[ ... \]
    math_pattern = re.compile(r"\\\((.*?)\\\)|\\\[(.*?)\\\]")

    # Check if the content contains any LaTeX math expressions
    if not math_pattern.search(content):
        # If no math expressions are found, render the content as plain text
        st.markdown(content)
        return

    # Split the content into text and math parts
    parts = math_pattern.split(content)

    for part in parts:
        if part is None:
            continue
        if math_pattern.match(f"\\({part}\\)") or math_pattern.match(f"\\[{part}\\]"):
            # Render as LaTeX
            st.latex(part.strip())
        else:
            # Render as regular text
            st.markdown(part.strip())

# Chat container to display messages
chat_container = st.container()
with chat_container:
    for message in st.session_state.message_log:
        with st.chat_message(message["role"]):
            if message["role"] == "ai":
                render_response_with_math(message["content"])  # Render AI messages with math support
            else:
                st.markdown(message["content"])  # Render user messages as plain text

# Chat input for user queries
user_input = st.chat_input("Enter your question about the document...")

if user_input:
    # Append user input to the message log
    st.session_state.message_log.append({"role": "user", "content": user_input})

    with st.spinner("🧠 Thinking..."):
        # Process the user query
        relevant_docs = find_related_documents(user_input)
        prompt_chain = build_prompt_chain(user_input, relevant_docs)
        ai_response = generate_ai_response(prompt_chain)

    # Append AI response to the message log
    st.session_state.message_log.append({"role": "ai", "content": ai_response})

    # Rerun the app to update the chat UI
    st.rerun()