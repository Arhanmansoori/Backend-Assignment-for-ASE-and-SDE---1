import itertools
from PyPDF2 import PdfReader
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import PDFDocument
from .serializers import PDFDocumentSerializer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai
import os
import tempfile

# API key management
API_KEYS = ["GEMINI API KEY ----- HERE  "]
api_key_cycle = itertools.cycle(API_KEYS)

def configure_api():
    api_key = next(api_key_cycle)
    genai.configure(api_key=api_key)
    return api_key

def extract_text_from_pdf(pdf_path):
    text = ""
    pdf_reader = PdfReader(pdf_path)
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def split_text_into_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
    return splitter.split_text(text)

def create_vector_store(chunks):
    api_key = configure_api()
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

class PDFUploadView(APIView):
    def post(self, request):
        """Handle multiple PDF uploads and process them together."""
        files = request.FILES.getlist('files')  # Get all uploaded files
        
        if not files:
            return Response({"error": "No files provided."}, status=status.HTTP_400_BAD_REQUEST)

        all_text_chunks = []

        with tempfile.TemporaryDirectory() as temp_dir:
            for file in files:
                # Save file temporarily
                temp_file_path = os.path.join(temp_dir, file.name)
                with open(temp_file_path, 'wb') as temp_file:
                    for chunk in file.chunks():
                        temp_file.write(chunk)
                
                # Save file record in the database
                pdf_document = PDFDocument.objects.create(file=file)
                
                # Process PDF
                text = extract_text_from_pdf(temp_file_path)
                text_chunks = split_text_into_chunks(text)
                all_text_chunks.extend(text_chunks)

                pdf_document.processed = True
                pdf_document.save()

        # Create vector store with combined chunks
        create_vector_store(all_text_chunks)

        return Response({"message": "PDFs uploaded and processed successfully."}, status=status.HTTP_201_CREATED)

class QuestionAnswerView(APIView):
    def post(self, request):
        question = request.data.get("question")
        if not question:
            return Response({"error": "Question is required."}, status=status.HTTP_400_BAD_REQUEST)

        api_key = configure_api()
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

        # Retrieve relevant documents
        docs = vector_store.similarity_search(question)

        # Define conversational chain
        prompt_template = """
        Use the provided context to answer the question comprehensively. If unsure, avoid making guesses.
        
        Context:
        {context}
        
        Question:
        {question}
        """
        model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=api_key, temperature=0.3)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)

        response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
        return Response({"answer": response['output_text']}, status=status.HTTP_200_OK)
