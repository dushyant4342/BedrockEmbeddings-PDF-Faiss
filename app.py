import json
import os
import sys
import boto3
import streamlit as st
import base64

## LangChain imports
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

## AWS Bedrock Clients
bedrock = boto3.client(service_name="bedrock-runtime", region_name="ap-south-1")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", client=bedrock)

# Default resume path
DEFAULT_RESUME_PATH = "data/DushyantCoverLetter.pdf"
VECTOR_STORE_PATH = "faiss_index"

## Load PDF, Split Text, and Generate Embeddings
def process_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.split_documents(documents)
    return docs

## Create or Load FAISS Vector Store
def get_vector_store(docs=None, recreate=False):
    if not os.path.exists(VECTOR_STORE_PATH) or recreate:
        if docs is None:
            docs = process_pdf(DEFAULT_RESUME_PATH)
        
        # Error handling for empty documents
        if not docs:
            #st.error("No documents found. Please upload a valid PDF.")
            return None

        try:
            vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
            vectorstore_faiss.save_local(VECTOR_STORE_PATH)
        except Exception as e:
            st.error(f"Error generating embeddings: {e}")
            return None

    return FAISS.load_local(VECTOR_STORE_PATH, bedrock_embeddings, allow_dangerous_deserialization=True)

## Load Llama3 Model
def get_llama3_llm():
    return Bedrock(model_id="meta.llama3-8b-instruct-v1:0", client=bedrock, model_kwargs={'max_gen_len': 512})

## Improved Prompt Template
prompt_template = """
Human: You are a helpful AI assistant that provides detailed and natural-sounding responses based on any asked question or the document context provided.

<context>
{context}
</context>

Question: {question}

Please follow these guidelines in your response:
1. Be conversational and engaging, avoid phrases like "based on the context" or "according to the document" and also avoid word "context" while answering
2. Provide detailed, thorough answers (at least 100-150 words when appropriate)
3. Include specific information from the context when relevant
4. If the question cannot be answered from the context, draw on your general knowledge to provide a helpful response
5. Format your response clearly with proper paragraphing and try to make answers as per your knowledge about anything like space, travel itinerary, maths, code debug anything

A:
"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

## Get Response from LLM
def get_response_llm(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa({"query": query})
    return answer['result']

## Custom CSS for enhanced UI
def apply_custom_css():
    st.markdown("""
    <style>
    .stApp {
        background-color: #121212;
        color: #FFFFFF;
    }
    .stTextInput > div > div > input {
        background-color: #2D2D2D;
        color: white;
        border-color: #444444;
    }
    .stTextInput > label {
        color: #BBBBBB !important;
    }
    .stSidebar {
        background-color: #1E1E1E;
    }
    .stSidebar .stFileUploader > div {
        background-color: #2D2D2D;
    }
    .stButton > button {
        background-color: #4B4B4B;
        color: white;
    }
    .stButton > button:hover {
        background-color: #606060;
    }
    .stTitle {
        font-weight: bold;
        color: #E0E0E0;
    }
    .footer-text {
        position: fixed;
        bottom: 20px;
        left: 20px;
        color: #888888;
        font-size: 0.8em;
        width: 300px;
        line-height: 1.5;
        background-color: rgba(18, 18, 18, 0.7);
        padding: 10px;
        border-radius: 5px;
        z-index: 1000;
        border-left: 2px solid #4CAF50;
    }
    .main-header {
        display: flex;
        align-items: center;
        background: linear-gradient(90deg, #1A1A1A, #303030);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        border-left: 5px solid #4CAF50;
    }
    .header-text {
        color: #FFFFFF;
        font-size: 1.8em;
        font-weight: bold;
    }
    .subheader-text {
        color: #BBBBBB;
        margin-top: 5px;
    }
    .chat-container {
        background-color: #1E1E1E;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
    }
    .stTextArea textarea {
        background-color: #2D2D2D;
        color: white;
        border-color: #444444;
        border-radius: 10px;
        padding: 10px;
        font-size: 16px;
        width: 100% !important;
        max-width: 1200px !important;
        box-sizing: border-box;
        min-height: 120px;
        line-height: 1.5;
    }
    .submit-button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        font-weight: bold;
        margin-top: 10px;
        transition: background-color 0.3s;
    }
    .submit-button:hover {
        background-color: #45a049;
    }
    .chat-input-container {
        display: flex;
        flex-direction: column;
        max-width: 1200px;
    }
    .upload-section {
        margin-bottom: 20px;
    }
    .powered-by-section {
        margin-top: 20px;
        color: #888888;
        font-size: 0.8em;
        line-height: 1.5;
    }
    </style>
    """, unsafe_allow_html=True)

## Display PDF in sidebar
def display_pdf(file_bytes, file_name):
    base64_pdf = base64.b64encode(file_bytes).decode("utf-8")
    pdf_display = f"""
    <iframe 
        src="data:application/pdf;base64,{base64_pdf}" 
        width="100%" 
        height="600px" 
        type="application/pdf"
    >
    </iframe>
    """
    st.markdown(f"### Preview of {file_name}")
    st.markdown(pdf_display, unsafe_allow_html=True)

## Streamlit App with Enhanced UI
def main():
    st.set_page_config("AI Document Chat", layout="wide")  
    apply_custom_css()

    st.title("🤖 Ask Anything or 📂 Upload a Document")
    st.write("Chat with AI about anything or upload your own PDF for analysis. ✨ By default, the web app provides information about Dushyant's professional profile, skills, and experiences.")
    
    # Custom Header
    st.markdown("""
    <div class="main-header">
        <div>
            <div class="header-text">How can I assist you today?</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize vector store
    get_vector_store(recreate=True)
    
    ## Sidebar: Upload New Resume with improved styling
    with st.sidebar:
        st.markdown("<h2 style='color: #E0E0E0;'>📄 Document Analysis</h2>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

        if uploaded_file is not None:
            temp_path = f"data/{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success("Uploaded successfully! Using this for Q&A.")
            docs = process_pdf(temp_path)
            get_vector_store(docs, recreate=True)
            display_pdf(uploaded_file.getbuffer(), uploaded_file.name)
        else:
            st.markdown("<p style='color: #BBBBBB;'> If no document is uploaded, the web app responds from Dushyant's cover letter.</p>", unsafe_allow_html=True)
            get_vector_store()

        # Powered by section below the upload section
        st.markdown("""
        <div class="powered-by-section">
            Powered by:<br>
            Amazon Titan Embedding Model<br>
            Meta Llama3 Text Generation<br>
            Hosted on AWS EC2 Instance<br>
            © 2025 All Rights Reserved
        </div>
        """, unsafe_allow_html=True)

    ## User Query with improved UI and Enter button
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("💬 Ask any question:"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            with st.spinner("🧠 Thinking..."):
                faiss_index = get_vector_store()
                llm = get_llama3_llm()
                response = get_response_llm(llm, faiss_index, prompt)
                st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()