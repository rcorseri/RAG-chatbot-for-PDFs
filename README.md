#  PDF Chat RAG Application

A command-line Retrieval-Augmented Generation (RAG) application that allows you to chat with your PDF documents using natural language. Built with Python and LangChain.

**Author:** Romain Corseri  
📧 **Contact:** romain.corseri@gmail.com

---

## 🎯 Overview

This project is a command-line RAG application coded in Python using LangChain open-source libraries to orchestrate all components of the application (Embedding model, Vector store, and LLM). The application enables natural language conversations with your PDF documents through an intelligent question-answering system.

## 🏗️ Architecture

- **🔗 Framework**: LangChain for component orchestration
- **🤗 Embeddings**: HuggingFace API for document vectorization  
- **🧠 LLM**: MistralAI for intelligent responses
- **📊 Vector Store**: In-memory storage with persistence
- **📄 PDF Processing**: PDFPlumber for robust document parsing

## ✨ Features

- 📚 **Multi-PDF Support**: Process multiple PDF documents simultaneously
- 🔍 **Semantic Search**: Advanced similarity-based document retrieval
- 💬 **Natural Language Chat**: Conversational interface in the terminal
- 💾 **Persistent Storage**: Process once, chat multiple times
- 🚀 **Fast Setup**: Simple configuration and execution
- 🎯 **Accurate Responses**: Context-aware answers from your documents

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Ubuntu/Linux terminal
- API tokens (see step 1)

### Step-by-Step Setup

#### 1. **Configure API Tokens**
Create a `.env` file with your API keys:
```bash
# Required tokens
MISTRAL_API_KEY=your_mistral_api_key_here
HUGGINGFACE_API_TOKEN=your_huggingface_token_here
LANGCHAIN_API_KEY=your_langchain_api_key_here 

#### 2. **Install dependencies**
```bash
pip install -r requirements.txt

#### 3. **Copy your PDF documents in the data folder**
```bash
mkdir data

#### 4. **Process your documents, create the embeddings and store in vector database**
```bash
mkdir vectordb
python ingest.py

#### 4. **Start chatting with your PDF documens in natural language**
```bash
python chat.py

