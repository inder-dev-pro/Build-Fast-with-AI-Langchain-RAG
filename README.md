# 📚 PDF RAG Q&A System

A powerful Retrieval-Augmented Generation (RAG) system that enables question answering over PDF documents using LangChain and Google's Gemini Pro.

## 🚀 Features

- 📄 Process multiple PDF documents simultaneously
- 🔍 Advanced text chunking with configurable overlap
- 🤖 Powered by Google's Gemini Pro for high-quality responses
- 📊 FAISS vector store for efficient similarity search
- 💡 Context-aware answers with source document references
- 🎯 User-friendly Streamlit interface

## 🛠️ Technical Architecture

The system is built using a modular architecture with the following components:

- **PDFProcessor**: Handles PDF text extraction and chunking
- **VectorStoreManager**: Manages FAISS vector store operations
- **QASystem**: Implements the RAG-based question answering
- **StreamlitApp**: Provides the user interface

## 📋 Prerequisites

- Python 3.7+
- Google Cloud API key with access to Gemini Pro
- Required Python packages (see requirements.txt)

## 🔧 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd pdf-rag-qa
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the root directory:
```
GOOGLE_API_KEY=your_api_key_here
```

## 🚀 Usage

1. Start the application:
```bash
streamlit run rag_pdf_qa.py
```

2. Access the web interface at `http://localhost:8501`

3. Upload your PDF documents using the sidebar

4. Process the documents by clicking "Process Documents"

5. Ask questions about the content of your documents

## 💻 How It Works

1. **Document Processing**:
   - PDFs are uploaded and processed
   - Text is extracted and split into chunks
   - Chunks are embedded and stored in FAISS

2. **Question Answering**:
   - User questions are processed
   - Relevant document chunks are retrieved
   - Gemini Pro generates context-aware answers

## 📦 Dependencies

- streamlit
- PyPDF2
- langchain
- langchain-google-genai
- faiss-cpu
- python-dotenv
- google-generativeai

## 🔒 Security Notes

- Never commit your `.env` file
- Keep your Google API key secure
- Don't expose API keys in client-side code

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Google Gemini Pro for the LLM capabilities
- LangChain for the RAG framework
- FAISS for efficient vector similarity search
- Streamlit for the web interface 