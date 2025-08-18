import os
import pickle
import glob
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

# Load environment variables
load_dotenv()

def ingest_multiple_pdfs(data_folder: str = "data", vector_store_path: str = "vectordb/vector_store_all.pkl"):
    """
    Ingest all PDF files in a folder and create a unified vector store.
    
    Args:
        data_folder (str): Path to the folder containing PDF files
        vector_store_path (str): Path to save the vector store
    """
    
    print(f"🔄 Starting batch ingestion from folder: {data_folder}")
    
    # Check if folder exists
    if not os.path.exists(data_folder):
        raise FileNotFoundError(f"Data folder not found: {data_folder}")
    
    # Find all PDF files in the folder
    pdf_pattern = os.path.join(data_folder, "*.pdf")
    pdf_files = glob.glob(pdf_pattern)
    
    if not pdf_files:
        print(f"⚠️  No PDF files found in {data_folder}")
        return None
    
    print(f"📄 Found {len(pdf_files)} PDF files:")
    for pdf_file in pdf_files:
        print(f"   - {os.path.basename(pdf_file)}")
    print()
    
    all_documents = []
    processed_files = []
    
    # Process each PDF file
    for i, pdf_path in enumerate(pdf_files, 1):
        try:
            filename = os.path.basename(pdf_path)
            print(f"📖 Processing file {i}/{len(pdf_files)}: {filename}")
            
            # Load PDF document
            loader = PDFPlumberLoader(pdf_path)
            docs = loader.load()
            
            # Add source filename to metadata for each document
            for doc in docs:
                doc.metadata["source_file"] = filename
            
            all_documents.extend(docs)
            processed_files.append(filename)
            
            print(f"   ✅ Loaded {len(docs)} pages from {filename}")
            
        except Exception as e:
            print(f"   ❌ Error processing {filename}: {str(e)}")
            continue
    
    if not all_documents:
        print("❌ No documents were successfully processed.")
        return None
    
    total_pages = len(all_documents)
    print(f"\n📊 Total pages loaded: {total_pages} from {len(processed_files)} files")
    
    # Split documents into chunks
    print("🔪 Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200, 
        add_start_index=True
    )
    all_splits = text_splitter.split_documents(all_documents)
    
    print(f"✅ Created {len(all_splits)} chunks from all documents")
    
    # Initialize embeddings model
    print("🤖 Initializing embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    
    # Create vector store and add documents
    print("🗄️ Creating vector store and computing embeddings...")
    vector_store = InMemoryVectorStore(embeddings)
    ids = vector_store.add_documents(documents=all_splits)
    
    print(f"✅ Added {len(ids)} document chunks to vector store")
    
    # Save vector store to disk
    print(f"💾 Saving vector store to: {vector_store_path}")
    with open(vector_store_path, 'wb') as f:
        pickle.dump(vector_store, f)
    
    print("\n🎉 Batch ingestion completed successfully!")
    print(f"📁 Processed files: {', '.join(processed_files)}")
    
    return vector_store

def ingest_pdf(file_path: str, vector_store_path: str = "vectordb/vector_store.pkl"):
    """
    Ingest a PDF file and create a vector store with embeddings.
    
    Args:
        file_path (str): Path to the PDF file
        vector_store_path (str): Path to save the vector store
    """
    
    print(f"🔄 Starting ingestion of: {file_path}")
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found: {file_path}")
    
    # Load PDF document
    print("📖 Loading PDF document...")
    loader = PDFPlumberLoader(file_path)
    docs = loader.load()
    
    print(f"✅ Loaded {len(docs)} pages from PDF")
    print(f"📄 Document preview: {docs[0].page_content[:100]}...")
    
    # Split documents into chunks
    print("🔪 Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200, 
        add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)
    
    print(f"✅ Created {len(all_splits)} chunks")
    
    # Initialize embeddings model
    print("🤖 Initializing embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    
    # Create vector store and add documents
    print("🗄️ Creating vector store and computing embeddings...")
    vector_store = InMemoryVectorStore(embeddings)
    ids = vector_store.add_documents(documents=all_splits)
    
    print(f"✅ Added {len(ids)} document chunks to vector store")
    
    # Save vector store to disk
    print(f"💾 Saving vector store to: {vector_store_path}")
    with open(vector_store_path, 'wb') as f:
        pickle.dump(vector_store, f)
    
    print("🎉 Ingestion completed successfully!")
    return vector_store

def load_vector_store(vector_store_path: str = "vector_store.pkl"):
    """
    Load a previously saved vector store.
    
    Args:
        vector_store_path (str): Path to the saved vector store
        
    Returns:
        InMemoryVectorStore: Loaded vector store
    """
    if not os.path.exists(vector_store_path):
        raise FileNotFoundError(f"Vector store not found: {vector_store_path}")
    
    print(f"📥 Loading vector store from: {vector_store_path}")
    with open(vector_store_path, 'rb') as f:
        vector_store = pickle.load(f)
    
    print("✅ Vector store loaded successfully!")
    return vector_store

if __name__ == "__main__":
    # Configuration
    DATA_FOLDER = "data"
    SINGLE_PDF_FILE = "data/4408_LOGS.pdf"
    VECTOR_STORE_PATH = "vectordb/vector_store_all.pkl"
    
    try:
        # Create data directory if it doesn't exist
        os.makedirs(DATA_FOLDER, exist_ok=True)
        
        # Check if vector store already exists
        if os.path.exists(VECTOR_STORE_PATH):
            response = input(f"Vector store '{VECTOR_STORE_PATH}' already exists. Overwrite? (y/n): ")
            if response.lower() != 'y':
                print("❌ Ingestion cancelled.")
                exit()
        
        # Ask user which mode to use
        print("🔧 Choose ingestion mode:")
        print("1. Single PDF file")
        print("2. All PDF files in data folder")
        
        while True:
            choice = input("Enter your choice (1 or 2): ").strip()
            if choice in ['1', '2']:
                break
            print("Invalid choice. Please enter 1 or 2.")
        
        if choice == '1':
            # Single file mode
            if not os.path.exists(SINGLE_PDF_FILE):
                print(f"❌ PDF file not found: {SINGLE_PDF_FILE}")
                print("Please place your PDF file in the data folder or update the SINGLE_PDF_FILE path.")
                exit(1)
            
            vector_store = ingest_pdf(SINGLE_PDF_FILE, VECTOR_STORE_PATH)
            print(f"\n📊 Vector store stats:")
            print(f"   - File processed: {SINGLE_PDF_FILE}")
            
        else:
            # Multiple files mode
            vector_store = ingest_multiple_pdfs(DATA_FOLDER, VECTOR_STORE_PATH)
            if vector_store is None:
                exit(1)
            
            print(f"\n📊 Vector store stats:")
            print(f"   - Folder processed: {DATA_FOLDER}")
        
        print(f"   - Vector store saved: {VECTOR_STORE_PATH}")
        print(f"   - Ready for chat! Run 'python chat.py' to start chatting with your document(s).")
        
    except Exception as e:
        print(f"❌ Error during ingestion: {str(e)}")
        exit(1)
