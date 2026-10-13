# AI Financial Research Assistant using RAG

An AI-powered financial document research application that allows users to upload annual reports and ask questions using Retrieval-Augmented Generation (RAG).

## Overview

The application extracts text from financial PDFs, splits documents into chunks, generates embeddings, and stores them in ChromaDB. When a user asks a question, the system retrieves relevant passages and uses a Large Language Model (LLM) to generate an evidence-based response.

## Features

- Upload financial reports in PDF format.
- Extract and chunk document text.
- Generate text embeddings using HuggingFace.
- Store and retrieve document chunks using ChromaDB.
- Ask questions about financial documents using natural language.
- Display retrieved source passages and page references.
- Use prompt instructions to reduce unsupported answers.

## Tech Stack

- Python
- LangChain
- Retrieval-Augmented Generation (RAG)
- HuggingFace Sentence Transformers
- ChromaDB
- Groq API
- Streamlit
- PyPDF

## Architecture

PDF Upload → Text Extraction → Chunking → Embeddings → ChromaDB → Similarity Search → LLM → Answer with Sources

## Installation

### 1. Clone the repository


git clone https://github.com/YOUR_USERNAME/financial-rag-assistant.git
cd financial-rag-assistant


### 2. Create a virtual environment

Windows:


py -m venv .venv
.venv\Scripts\activate


### 3. Install dependencies


python -m pip install -r requirements.txt


### 4. Configure the API key

Create a `.env` file in the project root:

```text
GROQ_API_KEY=your_actual_api_key
```

Get an API key from https://console.groq.com/.

Never commit your API key to GitHub.

### 5. Run the application


streamlit run app.py


## Example Questions

- Summarize the company's financial performance.
- What risks are mentioned in the annual report?
- What does the report say about revenue and expenses?
- Find the passages supporting a financial statement.

## Limitations

- Answers depend on the quality and readability of uploaded PDFs.
- Scanned documents may require OCR.
- Extracting financial tables accurately may require additional processing.
- LLM responses and generated citations should be verified against the original report.
- The application is intended for financial research and education, not personalized investment advice.

## Future Improvements

- Multi-document support and company report comparison.
- Financial table extraction and year-over-year calculations.
- Improved retrieval, reranking, and citation verification.
- RAG evaluation using Recall@k and answer-quality metrics.
- Automated testing and application deployment.

## Learning Outcomes

This project demonstrates practical understanding of RAG architecture, document processing, embeddings, vector databases, semantic retrieval, prompt engineering, and LLM application development.

