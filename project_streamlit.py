import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import itertools
import os
import tempfile
import google.generativeai as genai

API_KEYS = ["AIzaSyA7-lUzRsmTsocpAqsGs3-_F7-WYm8vIjE"]  
api_key_iterator = itertools.cycle(API_KEYS)

def initialize_genai_api():
    """Configure Google Generative AI with the next available API key."""
    key = next(api_key_iterator)
    genai.configure(api_key=key)
    return key

def extract_text_from_pdf(file_path):
    """Read and extract text content from a PDF file."""
    reader = PdfReader(file_path)
    return "".join(page.extract_text() or "" for page in reader.pages)

def split_text_into_chunks(content, chunk_size=7500, overlap=100):
    """Divide large text into manageable chunks for embedding."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return text_splitter.split_text(content)

def build_vector_store(file_paths):
    """Create and store vector embeddings for multiple PDF documents."""
    initialize_genai_api()
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=next(api_key_iterator))
    
    combined_chunks = []
    for file_path in file_paths:
        pdf_text = extract_text_from_pdf(file_path)
        chunks = split_text_into_chunks(pdf_text)
        combined_chunks.extend(chunks)
    
    vector_store = FAISS.from_texts(combined_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def setup_chat_chain():
    """Configure the chat model and return a conversational QA chain."""
    initialize_genai_api()
    chat_model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=next(api_key_iterator), temperature=0.3)
    
    prompt = PromptTemplate(
        template="""
        Use the provided context to answer the question comprehensively. If unsure, avoid making guesses.
        
        Context:
        {context}
        
        Question:
        {question}
        
        Answer:
        """,
        input_variables=["context", "question"]
    )
    return load_qa_chain(llm=chat_model, chain_type="stuff", prompt=prompt)

def query_knowledge_base(question):
    """Fetch relevant documents and use the conversational chain to generate an answer."""
    initialize_genai_api()
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=next(api_key_iterator))
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    matched_docs = vector_store.similarity_search(question)
    chat_chain = setup_chat_chain()
    response = chat_chain({"input_documents": matched_docs, "question": question}, return_only_outputs=True)
    return response

st.title("Multi-PDF AI Chat Assistant")
st.write("Upload multiple PDF documents to create a knowledge base and interact through questions.")

uploaded_pdfs = st.file_uploader("Select PDF files", type="pdf", accept_multiple_files=True)

if uploaded_pdfs:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_files = []
        for uploaded_file in uploaded_pdfs:
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())
            temp_files.append(temp_path)
        
        st.write("Preparing knowledge base from uploaded files...")
        build_vector_store(temp_files)
        st.success("Knowledge base ready! Ask any question based on the uploaded PDFs.")

        user_query = st.text_input("Type your question here:")
        if user_query:
            response = query_knowledge_base(user_query)
            st.write("### Answer:")
            st.write(response["output_text"])
