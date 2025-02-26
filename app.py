import json
import os
import boto3
import streamlit as st
import datetime
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# AWS Bedrock Client
bedrock = boto3.client(service_name="bedrock-runtime",region = 'ap-south-1')
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", client=bedrock)

# Default Resume PDF Path
DEFAULT_RESUME_PATH = "data/DushyantResume.pdf"

# Create Data Folder if not exists
os.makedirs("data", exist_ok=True)
os.makedirs("faiss_index", exist_ok=True)

# Streamlit Page Config
st.set_page_config(page_title="Dushyant's Career Chat", layout="wide", page_icon="💼")

# Sidebar: Upload PDF
with st.sidebar:
    st.title("📂 Upload a Document")
    uploaded_file = st.file_uploader("Upload a PDF (Optional)", type="pdf")
    
    if uploaded_file:
        temp_path = f"data/{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("✅ Document uploaded successfully! Using this for Q&A.")
        active_doc = temp_path
    else:
        active_doc = DEFAULT_RESUME_PATH

# Data Ingestion
def process_pdf(pdf_path):
    loader = PyPDFDirectoryLoader(os.path.dirname(pdf_path))
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_documents(documents)

# Vector Embedding Store
def get_vector_store(pdf_path=DEFAULT_RESUME_PATH, recreate=False):
    index_path = "faiss_index"
    if recreate or not os.path.exists(index_path):
        docs = process_pdf(pdf_path)
        vectorstore = FAISS.from_documents(docs, bedrock_embeddings)
        vectorstore.save_local(index_path)
    return FAISS.load_local(index_path, bedrock_embeddings, allow_dangerous_deserialization=True)

# Load FAISS Index
faiss_index = get_vector_store(active_doc, recreate=bool(uploaded_file))

# Llama3 Model
def get_llama3_llm():
    return Bedrock(model_id="meta.llama3-8b-instruct-v1:0", client=bedrock, model_kwargs={'max_gen_len': 512})

# Custom Prompt
PROMPT = PromptTemplate(
    template="""
    Human: Use the provided context to answer the question accurately. Summarize in at least 100 words with detailed explanations.
    
    <context>
    {context}
    </context>
    
    Question: {question}

    Assistant:""",
    input_variables=["context", "question"]
)

# Get LLM Response
def get_response_llm(llm, vectorstore, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa({"query": query})
    return answer['result']

# Initialize Chat History
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat UI
st.title("💼 Chat to know about Dushyant's Career or Upload new document 📄🤖")
st.write("Ask questions about **Dushyant's career, experience, skills, or any new document for summarization/classification**.")

# Chat Input
user_question = st.text_input("💬 Type your question and press Enter:", key="user_input")

if user_question:
    with st.spinner("Thinking... 🤔"):
        llm = get_llama3_llm()
        response = get_response_llm(llm, faiss_index, user_question)
        
        # Append to chat history
        st.session_state.chat_history.append({"question": user_question, "answer": response})

# Chat Display
st.write("### 📝 Chat History")
for chat in reversed(st.session_state.chat_history):  # Display latest first
    with st.chat_message("user"):
        st.markdown(f"**You:** {chat['question']}")
    with st.chat_message("assistant"):
        st.markdown(f"**🤖 AI:** {chat['answer']}")

# Clear Chat Button
if st.sidebar.button("🧹 Clear Chat History"):
    st.session_state.chat_history = []
    st.sidebar.success("Chat history cleared!")
