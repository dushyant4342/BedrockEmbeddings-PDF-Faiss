import json
import os
import sys
import boto3
import streamlit as st

## LangChain imports
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

## AWS Bedrock Clients
#bedrock = boto3.client(service_name="bedrock-runtime")
bedrock = boto3.client(service_name="bedrock-runtime",  region_name="ap-south-1")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", client=bedrock)

# Default resume path
DEFAULT_RESUME_PATH = "data/DushyantResume.pdf"
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
        vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
        vectorstore_faiss.save_local(VECTOR_STORE_PATH)
    return FAISS.load_local(VECTOR_STORE_PATH, bedrock_embeddings, allow_dangerous_deserialization=True)

## Load Llama3 Model
def get_llama3_llm():
    return Bedrock(model_id="meta.llama3-8b-instruct-v1:0", client=bedrock, model_kwargs={'max_gen_len': 512})

## Prompt Template
prompt_template = """
Human: Use the following context to answer the question concisely and in detail (min 100 words). Try to reply as good as possible don't answer like from the context. Talk naturally and be nice. If you don't find answer in the document try to think and then respond with good answers for any possible question could be anything realted to geography, Maths, Code debugging etc.

<context>
{context}
</context>

Question: {question}

Assistant:
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

## **Streamlit App**
def main():
    st.set_page_config("📄 PDF Vector Embedding", layout="wide")  
    st.title("💬 Chat with Dushyant or 📂 Upload a Document")  
    st.write("Upload a document for summarization, classification, or any other analysis. ✨ By default, this provides a summary of the Dushyant’s skills, achievements, work experience, education etc.")  

    get_vector_store(recreate=True)
    
    ## Sidebar: Upload New Resume
    with st.sidebar:
        st.title("Make summary of another document")
        uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

        if uploaded_file is not None:
            temp_path = f"data/{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success("Uploaded successfully! Using this for Q&A.")
            docs = process_pdf(temp_path)
            get_vector_store(docs, recreate=True)
        else:
            get_vector_store()  # Load default resume

    ## User Query
    user_question = st.text_input("Ask a question:")
    if user_question:
        with st.spinner("Thinking..."):
            faiss_index = get_vector_store()
            llm = get_llama3_llm()
            response = get_response_llm(llm, faiss_index, user_question)
            st.write(response)

if __name__ == "__main__":
    main()
