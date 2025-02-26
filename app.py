import os
import streamlit as st
import boto3
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# AWS Bedrock Client
bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")  # Update your region
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", client=bedrock)

# Persistent Storage Path on EC2 (Outside Docker)
STORAGE_PATH = "/home/ec2-user/data/"

# Ensure directories exist
os.makedirs(STORAGE_PATH, exist_ok=True)

PDF_PATH = os.path.join(STORAGE_PATH, "DushyantResume.pdf")  # Default resume file
INDEX_PATH = os.path.join(STORAGE_PATH, "faiss_index")  # FAISS Index Path


# **Load PDF & Generate FAISS Index**
def process_pdf(pdf_path):
    """Load PDF and split text for vector storage."""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_documents(documents)


def get_vector_store(pdf_path=PDF_PATH, recreate=False):
    """Load or create FAISS vector index."""
    if recreate or not os.path.exists(INDEX_PATH):
        st.info("Creating new FAISS index...")
        docs = process_pdf(pdf_path)
        vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
        vectorstore_faiss.save_local(INDEX_PATH)
    return FAISS.load_local(INDEX_PATH, bedrock_embeddings, allow_dangerous_deserialization=True)


# **LLM Model**
def get_llama3_llm():
    """Load Llama3 model from Bedrock."""
    return Bedrock(model_id="meta.llama3-8b-instruct-v1:0", client=bedrock, model_kwargs={'max_gen_len': 512})


# **Streamlit Web App**
def main():
    st.set_page_config("Career Q&A", layout="wide")
    
    st.title("🚀 Chat with Dushyant's Career Document")
    st.write("Ask any questions about Dushyant's career! (Default resume is used if no PDF is uploaded)")

    # **Sidebar for Upload**
    with st.sidebar:
        st.title("📂 Upload Your Document")
        uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

        if uploaded_file is not None:
            # Save file persistently in EC2
            uploaded_pdf_path = os.path.join(STORAGE_PATH, uploaded_file.name)
            with open(uploaded_pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.success("✅ File uploaded successfully! Using this for Q&A.")
            get_vector_store(uploaded_pdf_path, recreate=True)  # Update FAISS index

        else:
            get_vector_store()  # Load default resume

    # **Chat Interface**
    st.subheader("💬 Ask a Question")
    user_question = st.text_input("Type your question here...")

    if user_question:
        with st.spinner("Thinking... 🤔"):
            faiss_index = get_vector_store()
            llm = get_llama3_llm()
            response = get_response_llm(llm, faiss_index, user_question)

            # **Show Chat UI (Like ChatGPT)**
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []

            st.session_state.chat_history.append((user_question, response))

            # **Display Chat History**
            for question, answer in st.session_state.chat_history:
                with st.chat_message("user"):
                    st.write(f"**You:** {question}")
                with st.chat_message("assistant"):
                    st.write(f"**Bot:** {answer}")


# **Run the App**
if __name__ == "__main__":
    main()
