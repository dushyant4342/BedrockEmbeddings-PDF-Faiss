import os
import streamlit as st
import boto3
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA

# Set page configuration
st.set_page_config(
    page_title="💼 Career Q&A Bot",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4F8BF9;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .highlight {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.7rem;
        margin-bottom: 1rem;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.7rem;
        margin-bottom: 1rem;
        font-size: 1rem;
    }
    .user-message {
        background-color: #E3F2FD;
    }
    .bot-message {
        background-color: #F3F4F6;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# AWS Bedrock Client
@st.cache_resource
def get_bedrock_client():
    return boto3.client(service_name="bedrock-runtime", region_name="ap-south-1")

bedrock = get_bedrock_client()
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", client=bedrock)

# Paths - Inside Docker container
DATA_DIR = "/app/data/"  # This maps to EC2 path through Docker volume mount
DEFAULT_PDF = "DushyantResume.pdf"
PDF_PATH = os.path.join(DATA_DIR, DEFAULT_PDF)  # Default resume file
INDEX_PATH = os.path.join(DATA_DIR, "faiss_index")  # FAISS Index Path

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)

# Load PDF & Generate FAISS Index
def process_pdf(pdf_path):
    """Load PDF and split text for vector storage."""
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        return text_splitter.split_documents(documents)
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        st.error(f"Could not find or process file at: {pdf_path}")
        return []

@st.cache_resource
def get_vector_store(pdf_path=PDF_PATH, recreate=False):
    """Load or create FAISS vector index."""
    if recreate or not os.path.exists(INDEX_PATH):
        with st.spinner("🔍 Creating new document index... Please wait."):
            docs = process_pdf(pdf_path)
            if not docs:
                st.error("❌ No documents to process!")
                return None
            vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
            vectorstore_faiss.save_local(INDEX_PATH)
            st.success("✅ Document indexed successfully!")
    
    try:
        return FAISS.load_local(INDEX_PATH, bedrock_embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"Error loading index: {str(e)}")
        return None

# LLM Model
@st.cache_resource
def get_llm():
    """Load Llama3 model from Bedrock."""
    return Bedrock(
        model_id="meta.llama3-8b-instruct-v1:0", 
        client=bedrock, 
        region_name="ap-south-1",
        model_kwargs={'max_gen_len': 512, 'temperature': 0.7}
    )

def get_response_llm(llm, vectorstore, query):
    """Get response from LLM using retrieval QA."""
    if not vectorstore:
        return "Sorry, I couldn't access the document database. Please try uploading a document again."
        
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False
    )
    
    try:
        response = qa_chain.invoke({"query": query})
        return response['result']
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}"

def clear_chat():
    st.session_state.chat_history = []

# Streamlit Web App
def main():
    # Sidebar
    with st.sidebar:
        st.markdown("# 📂 Document Upload")
        st.markdown("---")
        
        uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
        
        if uploaded_file is not None:
            st.success(f"✅ {uploaded_file.name} uploaded!")
            # Save file to data directory (which is mounted to EC2)
            uploaded_pdf_path = os.path.join(DATA_DIR, uploaded_file.name)
            with open(uploaded_pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.info("📊 Processing document...")
            get_vector_store(uploaded_pdf_path, recreate=True)  # Update FAISS index
            current_doc = uploaded_file.name
        else:
            current_doc = DEFAULT_PDF
            get_vector_store()  # Load default resume
        
        st.markdown("---")
        st.markdown(f"🔍 **Current document**: {current_doc}")
        st.markdown("---")
        
        if st.button("🗑️ Clear Chat History"):
            clear_chat()
            st.success("Chat history cleared!")
        
        st.markdown("---")
        st.markdown("### 🧠 About")
        st.markdown("""
        This app uses AWS Bedrock with:
        - Titan Embeddings
        - Llama 3 8B
        - FAISS vector search
        """)

    # Main area
    st.markdown('<h1 class="main-header">💼 Career Document Assistant</h1>', unsafe_allow_html=True)
    
    st.markdown('<div class="highlight">', unsafe_allow_html=True)
    st.markdown("""
    👋 Welcome! I can help you explore career documents and answer questions about them.
    
    By default, I'll answer questions about Dushyant's career document, but you can upload any PDF using the sidebar.
    
    Some example questions:
    - What are Dushyant's key skills?
    - Tell me about the most recent work experience
    - What educational background do they have?
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat interface
    st.markdown('<h2 class="sub-header">💬 Chat</h2>', unsafe_allow_html=True)
    
    # Display chat history
    for question, answer in st.session_state.chat_history:
        st.markdown(f'<div class="chat-message user-message">🧑‍💼 <b>You:</b> {question}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="chat-message bot-message">🤖 <b>Assistant:</b> {answer}</div>', unsafe_allow_html=True)
    
    # Input area
    user_question = st.text_input("Ask a question about the document:", key="user_input", placeholder="Type your question here...")
    
    if user_question and user_question.strip():
        with st.spinner("🤔 Thinking..."):
            llm = get_llm()
            faiss_index = get_vector_store()
            response = get_response_llm(llm, faiss_index, user_question)
            
            # Add to chat history
            st.session_state.chat_history.append((user_question, response))
            
            # Rerun to update the display
            st.rerun()

if __name__ == "__main__":
    main()
