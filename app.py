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
bedrock = boto3.client(service_name="bedrock-runtime", region_name="ap-south-1")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", client=bedricht)

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

## Improved Prompt Template
prompt_template = """
Human: You are a helpful AI assistant that provides detailed and natural-sounding responses based on any asked question or the document context provided.

<context>
{context}
</context>

Question: {question}

Please follow these guidelines in your response:
1. Be conversational and engaging, avoid phrases like "based on the context" or "according to the document"
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
        background: linear-gradient(135deg, #121212, #1E1E1E);
        color: #FFFFFF;
    }
    .stTextInput > div > div > input, .stTextArea textarea {
        background-color: #2D2D2D;
        color: white;
        border-color: #444444;
        border-radius: 10px;
        padding: 10px;
        font-size: 16px;
    }
    .stTextInput > label, .stTextArea > label {
        color: #BBBBBB !important;
    }
    .stSidebar {
        background: linear-gradient(135deg, #1E1E1E, #2D2D2D);
    }
    .stSidebar .stFileUploader > div {
        background-color: #3A3A3A;
    }
    .stButton > button {
        background: linear-gradient(90deg, #4CAF50, #388E3C);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #45A049, #368033);
    }
    .stTitle {
        font-weight: bold;
        color: #E0E0E0;
    }
    .chat-container {
        background-color: rgba(30, 30, 30, 0.8);
        border-radius: 15px;
        padding: 30px;
        margin-top: 30px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
    }
    .chat-input-container {
        display: flex;
        flex-direction: column;
    }
    .submit-button {
        background: linear-gradient(90deg, #4CAF50, #388E3C);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 12px 25px;
        font-weight: bold;
        align-self: flex-end;
    }
    .submit-button:hover {
        background: linear-gradient(90deg, #45A049, #368033);
    }
    .response-box {
        background-color: #2D2D2D;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin-top: 20px;
    }
    .upload-section {
        background-color: rgba(45, 45, 45, 0.8);
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .upload-section h3 {
        color: #E0E0E0;
    }
    .powered-by-section {
        margin-top: 20px;
        color: #888888;
        font-size: 0.8em;
        line-height: 1.5;
    }
    </style>
    """, unsafe_allow_html=True)


## Streamlit App with Enhanced UI
def main():
    st.set_page_config("AI Document Chat", layout="wide")
    apply_custom_css()

    st.title("🤖 Amazon Titan-Inspired AI Assistant")
    st.write("Ask me anything or upload a document! ✨")

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
        else:
            st.markdown("<p style='color: #BBBBBB;'>Using default document.</p>", unsafe_allow_html=True)
            get_vector_store()

        st.markdown("""
        <div class="powered-by-section">
            Powered by:<br>
            Amazon Titan Embedding Model<br>
            Meta Llama3 Text Generation<br>
            Hosted on AWS EC2 Instance<br>
            © 2025 All Rights Reserved
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="chat-container">
        <div class="chat-input-container">
            <div class="upload-section">
                <h3>Upload Document</h3>
                <p>Upload a PDF to analyze its content.</p>
                <div style="display: none;">
                    <input type="file" id="file-upload" accept=".pdf">
                </div>
                <label for="file-upload" style="cursor: pointer; background: linear-gradient(90deg, #4CAF50, #388E3C); color: white; padding: 10px 20px; border-radius: 5px; display: inline-block;">Choose File</label>
            </div>
            <div class="chat-input-area">
                <textarea id="user-input" placeholder="Type your question here..." style="width: 100%; min-height: 150px; background-color: #2D2D2D; color: white; border: 1px solid #444444; border-radius: 10px; padding: 10px; font-size: 16px;"></textarea>
                <button class="submit-button" onclick="submitQuery()">Submit</button>
            </div>
        </div>
        <div id="response-area" class="response-box" style="display: none;"></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <script>
    function submitQuery() {
        var userInput = document.getElementById('user-input').value;
        if (userInput) {
            document.getElementById('response-area').style.display = 'block';
            document.getElementById('response-area').innerHTML = '<p>Thinking...</p>'; // Placeholder while processing
            sendQueryToStreamlit(userInput);
        }
    }

    function sendQueryToStreamlit(query) {
        // Use Streamlit's session state to send the query
        Streamlit.setComponentValue(query);
    }

    document.getElementById('file-upload').addEventListener('change', function(event) {
        var file = event.target.files[0];
        if (file) {
            // Handle file upload logic here (e.g., using st.file_uploader)
            // For now, just show the file name
            alert('File selected: ' + file.name);
        }
    });
    </script>
    """, unsafe_allow_html=True)

    # Streamlit component to receive query from JavaScript
    user_question = st.text_area("", key="query_input", height=0, label_visibility="hidden")

    if user_question:
        with st.spinner("Thinking..."):
            faiss_index = get_vector_store()
            llm = get_llama3_llm()
            response = get_response_llm(llm, faiss_index, user_question)
            st.markdown(f"""
            <script>
                document.getElementById('response-area').innerHTML = `{response.replace("`", "\\`")}`;
            </script>
            """, unsafe_allow_html=True)
            st.rerun()

if __name__ == "__main__":
    main()
