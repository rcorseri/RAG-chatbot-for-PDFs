import os
import sys
from datetime import datetime
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage
from itertools import chain
from ingest_all import load_vector_store

# Load environment variables
load_dotenv()

class DocumentChatBot:
    def __init__(self, vector_store_path: str = "vectordb/vector_store_all.pkl"):
        """Initialize the chatbot with a pre-built vector store."""
        try:
            self.vector_store = load_vector_store(vector_store_path)
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 10},
            )
            self.llm = init_chat_model("mistral-large-latest", model_provider="mistralai")
            print("🤖 Subsurface Document Assistant ready!\n")
        except FileNotFoundError:
            print("❌ Vector store not found. Please run 'python ingest.py' first to process your document.")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error initializing chatbot: {str(e)}")
            sys.exit(1)
    
    def get_system_prompt(self):
        """Enhanced system prompt for travel and event/conference document analysis."""
        return """You are a highly qualified personal assistance with expertise in:

🔍 **travel planning:**

🔍 **Conference schedule:**

📋 **Response Guidelines:**
- Provide precise, technical answers based strictly on the document content
- When citing measurements, time schedules, include exact values with units 
- For coordinate data, specify the reference system if available
- Explain technical terms when they might be unclear
- If multiple interpretations exist, present them clearly
- Always indicate your confidence level in the answer

⚠️ **Important Instructions:**
- If information is not in the provided context, clearly state "This information is not available in the document"
- For ambiguous queries, ask for clarification
- When discussing depths, always specify whether it's Measured Depth (MD), True Vertical Depth (TVD), or Total Depth (TD)
- Reference specific sections of the document when possible

🎯 **Answer Structure:**
1. Direct answer to the question
2. Supporting details from the document

Use the provided context to answer questions accurately and professionally."""

    def search_and_respond(self, question: str):
        """Search the document and generate a response."""
        print("🔍 Searching document...")
        
        # Retrieve relevant chunks
        context = self.retriever.batch([question])
        context_docs = list(chain.from_iterable(context))
        
        if not context_docs:
            return "❌ No relevant information found in the document for your question."
        
        # Prepare context text
        context_text = "\n".join(doc.page_content for doc in context_docs)
        
        # Build messages for the LLM
        messages = [
            SystemMessage(content=self.get_system_prompt()),
            HumanMessage(content=f"""**Document Context:**
{context_text}

**Question:** {question}

Please provide a comprehensive answer based on the document content above.""")
        ]
        
        # Generate response
        print("🧠 Analyzing and generating response...\n")
        response = self.llm.invoke(messages)
        
        return response.content
    
    def display_welcome(self):
        """Display welcome message and usage instructions."""
        print("="*80)
        print("🛢️PERSONAL ASSISTANT - EVENT PLANNER - TRAVEL AGENT")
        print("="*80)
        print("""
Welcome! I'm your specialized assistant in travel and conference planning.

🔍 **travel planning:**

🔍 **Conference schedule:**

📋 **Response Guidelines:**
- Provide precise, technical answers based strictly on the document content
- When citing measurements, time schedules, include exact values with units 
- For coordinate data, specify the reference system if available
- Explain technical terms when they might be unclear
- If multiple interpretations exist, present them clearly
- Always indicate your confidence level in the answer

⌨️  **Commands:**
• Type 'quit', 'exit', or 'bye' to end the session
• Type 'help' for this message again
• Just ask your question naturally!

""")
        print("="*80)
    
    def run_chat(self):
        """Main chat loop."""
        self.display_welcome()
        
        while True:
            try:
                # Get user input
                question = input("🔍 Your question: ").strip()
                
                if not question:
                    continue
                
                # Handle special commands
                if question.lower() in ['quit', 'exit', 'bye']:
                    print("\n👋 Goodbye! Thanks for using the Assistant / travel agent!")
                    break
                elif question.lower() == 'help':
                    self.display_welcome()
                    continue
                
                # Process the question
                #print(f"\n📝 Question: {question}")
                print("-" * 60)
                
                response = self.search_and_respond(question)
                
                print(f"🤖 **Answer:**")
                print(response)
                print("\n" + "="*80 + "\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye! Thanks for using the Subsurface Document Assistant!")
                break
            except Exception as e:
                print(f"❌ An error occurred: {str(e)}")
                print("Please try again or type 'quit' to exit.\n")

def main():
    """Main function to run the chat interface."""
    # Check if required environment variables are set
    if not os.getenv("MISTRAL_API_KEY"):
        print("⚠️  Warning: MISTRAL_API_KEY not found in environment variables.")
        print("Please set your Mistral API key in your .env file or environment.")
    
    # Initialize and run the chatbot
    try:
        chatbot = DocumentChatBot()
        chatbot.run_chat()
    except Exception as e:
        print(f"❌ Failed to start chatbot: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
