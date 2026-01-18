import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community. document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from pinecone import Pinecone, ServerlessSpec

# Load environment variables from .env file
load_dotenv()

# Initialize Pinecone

def initialize_pinecone():
    """Connect to Pinecone with your API key"""
    api_key = os.getenv("PINECONE_API_KEY")
    
    if not api_key:  
        print("\n ERROR:PINECONE_API_KEY not found!")
        raise ValueError("PINECONE_API_KEY not found")
    
    return Pinecone(api_key=api_key)

# Create Pinecone Index

def create_index(pc):
    """Create a new Pinecone index if it doesn't exist"""
    index_name = "techcorp-policies"
    
    # Check if index already exists
    existing_indexes = pc.list_indexes().names()
    
    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    
    return index_name

# Load Documents from Knowledge Base

def load_documents_from_folder(folder_path="knowledge_base"):
    """Read all . txt files from the knowledge_base folder"""
    documents = []
    
    if not Path(folder_path).exists():
        raise FileNotFoundError(f"Folder '{folder_path}' not found")
    
    for file_path in sorted(Path(folder_path).glob("*.txt")):
        loader = TextLoader(str(file_path), encoding="utf-8")
        docs = loader.load()
        
        for doc in docs:
            doc.metadata["source"] = file_path.name
        
        documents.extend(docs)
    
    return documents


# Split Documents into Chunks

def split_documents(documents):
    """Break documents into smaller chunks for better retrieval"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    return chunks


# Initialize Embeddings Model

def get_embeddings():
    """Create embeddings (convert text to numbers)"""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L12-v2"
    )


# Store Documents in Pinecone

def store_in_pinecone(chunks, embeddings, index_name):
    """Convert documents to embeddings and store in Pinecone"""
    vector_store = PineconeVectorStore. from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=index_name
    )
    
    return vector_store

# Create Retriever

def create_retriever(vector_store):
    """Create a retriever that finds relevant documents"""
    return vector_store.as_retriever(
        search_kwargs={"k": 3}
    )


# Format Retrieved Documents

def format_documents(docs):
    """Format documents with source filenames"""
    formatted_text = "\n\n---\n\n".join(
        f"Source: {doc. metadata['source']}\nContent:\n{doc.page_content}"
        for doc in docs
    )
    return formatted_text


# Initialize Groq LLM

def get_llm():
    """Initialize the Groq language model"""
    api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key:  
        print("\n ERROR:  GROQ_API_KEY not found!")
        raise ValueError("GROQ_API_KEY not found")
    
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.2 #keeps the llm responses more factual based 
    )

# Create Prompt Template

def create_prompt():
    """Create the prompt that guides the LLM"""
    prompt = ChatPromptTemplate. from_template("""You are a TechCorp HR Assistant. 

Your job is to answer employee questions based ONLY on the official company policies provided below.

STRICT RULES YOU MUST FOLLOW:
1. Use ONLY the most recent policy (2024 overrides 2021)
2. Ignore any documents that are NOT company policies (e.g., cafeteria menus)
3. ALWAYS cite the exact source filename in parentheses at the end
4. Start your answer with a clear YES or NO,
moreover if there is a conflict between the two(YES AND NO) then state the conditions as well.
5. Provide a concise explanation referencing the specific policy details
6. Be direct and factual - no unnecessary words

Policy Documents:
{context}

Question: {question}

Answer (format: Yes/No + reason + source filename):""")
    return prompt


# Build RAG Chain

def build_rag_chain(retriever, prompt, llm):
    """complete RAG pipeline """
    
    chain = (
        RunnableParallel(
            context=retriever | format_documents,
            question=RunnablePassthrough()
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain


# Chatbot with Memory

class TechCorpChatbot: 
    def __init__(self, chain):
        self.chain = chain
        self.conversation_memory = []
    
    def add_to_memory(self, user_message, ai_response):
        """Add user message and AI response to memory"""
        self.conversation_memory.append({
            "role": "user",
            "content": user_message
        })
        self.conversation_memory.append({
            "role": "assistant",
            "content": ai_response
        })
    
    def chat(self, user_message):
        """Get response from chatbot"""
        response = self.chain. invoke(user_message)
        self.add_to_memory(user_message, response)
        return response
    
    def display_welcome(self):
        """Display welcome message"""
        print("\n" + "=" * 60)
        print("TechCorp HR Assistant Chatbot")
        print("=" * 60)
        print("\nHello! I'm your TechCorp HR Assistant.")
        print("I can answer questions about company policies.")
        print("Type 'exit' to quit.\n")


# Initialize and Run Chatbot

def initialize_system():
    """Initialize all components and return chatbot"""
    pc = initialize_pinecone()
    index_name = create_index(pc)
    
    documents = load_documents_from_folder("knowledge_base")
    chunks = split_documents(documents)
    
    embeddings = get_embeddings()
    vector_store = store_in_pinecone(chunks, embeddings, index_name)
    
    retriever = create_retriever(vector_store)
    
    llm = get_llm()
    prompt = create_prompt()
    
    chain = build_rag_chain(retriever, prompt, llm)
    
    chatbot = TechCorpChatbot(chain)
    return chatbot


# Main Chatbot Loop

def main():
    """Run the chatbot"""
    chatbot = initialize_system()
    chatbot.display_welcome()
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() == "exit":
                print("\nAssistant: Thank you for using TechCorp HR Assistant. Goodbye!")
                break
            
            if not user_input:
                continue
            
            response = chatbot.chat(user_input)
            print(f"\nAssistant:  {response}\n")
        
        except KeyboardInterrupt: 
            print("\n\nAssistant: Thank you for using TechCorp HR Assistant. Goodbye!")
            break
        except Exception as e: 
            print(f"Error: {str(e)}")
            print("Please try again.\n")


# Run the Script
if __name__ == "__main__": 
    main()
