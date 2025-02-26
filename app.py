import os
import streamlit as st
import boto3
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.llms import Bedrock
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA

# Set page configuration
st.set_page_config(page_title="Career Q&A Bot", page_icon="📄", layout="wide")

# Custom CSS for better visibility
st.markdown(
    """
    <style>
        .stTextInput>div>div>input {
            font-size: 1.2rem;
            padding: 10px;
        }
        .stChatMessage {
            font-size: 1.2rem;
        }
        .bot-message {
            background-color: #f3f4f6;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
        }
        .user-message {
            background-color: #e3f2fd;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# AWS Bedrock Client
@st.cache_resource
def get_bedrock_client():
    return boto3.client(service_name="bedrock-runtime", region_name="ap-south-1")

bedrock = get_bedrock_client()
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", client=bedrock)

# Paths
DATA_DIR = "./data/"
DEFAULT_PDF = "DushyantResume.pdf"
PDF_PATH = os.path.join(DATA_DIR, DEFAULT_PDF)
INDEX_PATH = os.path.join(DATA_DIR, "faiss_index")

os.makedirs(DATA_DIR, exist_ok=True)

# Load PDF & Generate FAISS Index
@st.cache_resource
def process_and_store_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    
    if not docs:
        st.error("No valid text found in PDF.")
        return None
    
    vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    vectorstore_faiss.save_local(INDEX_PATH)
    return vectorstore_faiss

@st.cache_resource
def get_vector_store():
    if os.path.exists(INDEX_PATH):
        return FAISS.load_local(INDEX_PATH, bedrock_embeddings, allow_dangerous_deserialization=True)
    return process_and_store_pdf(PDF_PATH)

# Load LLM
@st.cache_resource
def get_llm():
    return Bedrock(
        model_id="meta.llama3-8b-instruct-v1:0", 
        client=bedrock, 
        region_name="ap-south-1",
        model_kwargs={'max_gen_len': 512, 'temperature': 0.7}
    )

llm = get_llm()
faiss_index = get_vector_store()

def get_response(query):
    if not faiss_index:
        return "Error: No document index found. Please upload a document."
    
    retriever = faiss_index.as_retriever(search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    
    try:
        response = qa_chain.invoke({"query": query})
        return response.get('result', "Error generating response.")
    except Exception as e:
        return f"Error: {str(e)}"

def clear_chat():
    st.session_state.chat_history = []

# Sidebar
with st.sidebar:
    st.header("📂 Document Upload")
    uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
    
    if uploaded_file:
        uploaded_pdf_path = os.path.join(DATA_DIR, uploaded_file.name)
        with open(uploaded_pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Uploaded {uploaded_file.name}")
        process_and_store_pdf(uploaded_pdf_path)
        faiss_index = get_vector_store()
    
    if st.button("🗑️ Clear Chat History"):
        clear_chat()
        st.success("Chat history cleared!")

# Main UI
st.title("💼 Career Document Assistant")
st.markdown("Ask questions about the uploaded document.")

# Display chat history
for question, answer in st.session_state.chat_history:
    st.markdown(f'<div class="user-message">🧑‍💼 <b>You:</b> {question}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="bot-message">🤖 <b>Assistant:</b> {answer}</div>', unsafe_allow_html=True)

# User input
user_question = st.text_input("Ask a question:", key="user_input")

if user_question.strip():
    response = get_response(user_question)
    st.session_state.chat_history.append((user_question, response))
    st.rerun()
