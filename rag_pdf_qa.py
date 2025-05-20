"""
PDF Question Answering System using RAG (Retrieval-Augmented Generation)
This application uses LangChain and Google's Gemini Pro to create a RAG-based Q&A system for PDF documents.
"""

import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=GOOGLE_API_KEY)

class PDFProcessor:
    """Handles PDF document processing and text extraction."""
    
    @staticmethod
    def extract_text(pdf_docs):
        """Extract text from multiple PDF documents."""
        text = ""
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text

    @staticmethod
    def create_chunks(text, chunk_size=2000, chunk_overlap=500):
        """Split text into chunks for processing."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return text_splitter.split_text(text)

class VectorStoreManager:
    """Manages vector store operations using FAISS."""
    
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    def create_vector_store(self, chunks):
        """Create and save vector store from text chunks."""
        vector_store = FAISS.from_texts(chunks, embedding=self.embeddings)
        vector_store.save_local("faiss_index")
        return len(vector_store.index_to_docstore_id)
    
    def load_vector_store(self):
        """Load existing vector store."""
        return FAISS.load_local("faiss_index", self.embeddings, allow_dangerous_deserialization=True)

class QASystem:
    """Handles question answering using RAG."""
    
    def __init__(self):
        self.chain = self._create_qa_chain()
    
    def _create_qa_chain(self):
        """Create the QA chain with custom prompt."""
        prompt_template = """
        Answer the question as detailed as possible from the given context. 
        If the answer is not in the context, say so and provide a general answer based on your knowledge.
        
        Context:
        {context}
        
        Question:
        {question}
        
        Answer:
        """
        
        model = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0.3,
            verbose=True
        )
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        return load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    
    def get_answer(self, question, vector_store):
        """Get answer for a question using RAG."""
        docs = vector_store.similarity_search(question, k=5)
        response = self.chain(
            {"input_documents": docs, "question": question},
            return_only_outputs=True
        )
        return response["output_text"]

class StreamlitApp:
    """Streamlit application interface."""
    
    def __init__(self):
        self.pdf_processor = PDFProcessor()
        self.vector_store_manager = VectorStoreManager()
        self.qa_system = QASystem()
    
    def setup_page(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="PDF RAG Q&A System",
            page_icon="📚",
            layout="wide"
        )
        st.title("📚 PDF Question Answering System")
    
    def process_pdfs(self, pdf_docs):
        """Process uploaded PDF documents."""
        with st.spinner("Processing PDFs..."):
            raw_text = self.pdf_processor.extract_text(pdf_docs)
            text_chunks = self.pdf_processor.create_chunks(raw_text)
            doc_count = self.vector_store_manager.create_vector_store(text_chunks)
            st.success(f"✅ Processed {doc_count} document chunks")
    
    def handle_question(self, question):
        """Handle user questions and display answers."""
        if not question.strip():
            st.warning("⚠️ Please enter a valid question!")
            return
        
        vector_store = self.vector_store_manager.load_vector_store()
        answer = self.qa_system.get_answer(question, vector_store)
        
        st.subheader("📢 AI Response:")
        st.success(answer)
    
    def run(self):
        """Run the Streamlit application."""
        self.setup_page()
        
        with st.sidebar:
            st.title("📂 Upload PDFs")
            pdf_docs = st.file_uploader(
                "Upload your PDF documents",
                type=["pdf"],
                accept_multiple_files=True
            )
            if st.button("Process Documents"):
                if pdf_docs:
                    self.process_pdfs(pdf_docs)
                else:
                    st.warning("⚠️ Please upload at least one PDF document!")
        
        st.subheader("💬 Ask Questions About Your Documents")
        user_question = st.text_input("Type your question below:")
        if st.button("Get Answer"):
            self.handle_question(user_question)

def main():
    """Main entry point of the application."""
    app = StreamlitApp()
    app.run()

if __name__ == "__main__":
    main()