# Streamlit frontend for a RAG implementation using Qdrant and LangChain

import streamlit as st
import os
from typing import List, Dict, Any, Optional
import pandas as pd
import re
import numpy as np
from tqdm.auto import tqdm

# LangSmith imports for tracing
from langsmith import traceable
from langchain.callbacks.tracers import LangChainTracer
from langchain.callbacks.manager import CallbackManager

# LangChain imports
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import Qdrant
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest


# Function to extract game name from query
def extract_game_name(query: str, games_csv_path: str = "games.csv") -> Optional[str]:
    """
    Extract game name from a user query by checking against a CSV file of game names.
    
    Args:
        query: The user's query text
        games_csv_path: Path to CSV file containing game names
        
    Returns:
        The extracted game name or None if no game found
    """
    # Check if CSV file exists
    if not os.path.exists(games_csv_path):
        print(f"Warning: Games CSV file {games_csv_path} not found.")
        return None
    
    try:
        # Load game names from CSV
        games_df = pd.read_csv(games_csv_path, header=None)
        game_names = games_df[0].tolist()  # Assuming the game names are in the first column
        
        # Normalize query for matching
        normalized_query = query.lower()
        
        # Sort game names by length (descending) to prioritize longer matches
        for game in sorted(game_names, key=len, reverse=True):
            # Check for case-insensitive match
            if game.lower() in normalized_query:
                return game
        
        # Enhanced detection with patterns
        patterns = [
            r"for\s+(.+?)(?:\s+game|\s+reviews|\s+in|\?|$)",  # "for [game]"
            r"about\s+(.+?)(?:\s+game|\s+reviews|\?|$)",      # "about [game]"
            r"in\s+(.+?)(?:\s+game|\s+reviews|\?|$)",         # "in [game]"
        ]
        
        for pattern in patterns:
            matches = re.search(pattern, normalized_query)
            if matches:
                potential_game = matches.group(1).strip()
                # Find the closest match in our game list
                for game in game_names:
                    if potential_game in game.lower() or game.lower() in potential_game:
                        return game
        
        return None
        
    except Exception as e:
        print(f"Error extracting game name: {e}")
        return None


# Initialize callback manager for tracing
def get_tracer_callback_manager():
    """Set up LangChain callback manager for tracing."""
    tracer = LangChainTracer(project_name=os.environ.get("LANGCHAIN_PROJECT", "steam-reviews-rag"))
    return CallbackManager([tracer])


# Setup Qdrant-based retriever
@traceable(name="setup_rag_retriever")
def setup_rag_retriever(
    collection_name: str = "steam_reviews",
    openai_api_key: str = None,
    embedding_model: str = "text-embedding-3-small",
    search_type: str = "similarity",
    k: int = 4
):
    """
    Set up a langchain retriever from an existing Qdrant collection with tracing.
    No game filtering - will format game name in the documents during context preparation.
    """
    if openai_api_key is None:
        openai_api_key = os.environ.get("OPENAI_API_KEY")
    
    # Setup embeddings
    embeddings = OpenAIEmbeddings(
        model=embedding_model,
        openai_api_key=openai_api_key
    )
    
    # Connect to existing Qdrant collection
    client = QdrantClient(host="localhost", port=6333)
    
    # Create vector store without game filtering
    vector_store = Qdrant(
        client=client,
        collection_name=collection_name,
        embeddings=embeddings,
    )
    
    # Create retriever with specified search parameters
    retriever = vector_store.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k}
    )
    
    return retriever


# Custom function to format documents
def format_documents(docs):
    """
    Format each document to include game name from metadata.
    Format: "Review for {game_name}: {review_content}"
    """
    formatted_docs = []
    for doc in docs:
        game_name = doc.metadata.get("game_name", "Unknown Game")
        formatted_text = f"Review for \"{game_name}\": {doc.page_content}"
        formatted_docs.append(formatted_text)
    return "\n\n".join(formatted_docs)


# Function to run RAG query with metrics
@traceable(name="run_rag_query")
def run_rag_query(qa_system, query, chat_history=None, previous_docs=None, reuse_context=False):
    """
    Run a RAG query with comprehensive tracing for performance analysis
    """
    # Prepare parameters
    params = {
        "query": query,
        "previous_docs": previous_docs,
        "reuse_context": reuse_context
    }
    if chat_history is not None:
        params["chat_history"] = chat_history
    
    # Execute query using invoke method
    result = qa_system.invoke(params)
    
    # Calculate result metrics
    run_metrics = {
        "query": query,
        "result_length": len(result.get("result", "")),
        "num_source_docs": len(result.get("source_documents", [])),
        "reused_context": reuse_context and previous_docs is not None
    }
    
    return result, run_metrics


# Updated RAG system setup with chat memory support
@traceable(name="setup_rag_system")
def setup_rag_system(collection_name="steam_reviews", k=25, model="gpt-4o-mini", temperature=0):
    """
    Set up RAG system that formats review context to include game name from metadata.
    Enhanced with support for chat history and context reuse.
    """
    # Set up client
    client = QdrantClient(host="localhost", port=6333)
    
    # Set up retriever without game filtering
    retriever = setup_rag_retriever(
        collection_name=collection_name, 
        k=k
    )
    
    # Set up prompt template with chat history support
    template = """You are an AI assistant helping game developers improve their games based on Steam reviews.
    To answer the question, you will be given a list of reviews mentioning the game requested. The reviews are formatted as follows:
    "Review for [game_name]: [review_content]"
    Always check that the game name is correct for the reviews considered. If the game requested does not match the game passed in one 
    or more reviews then ignore the reviews that do not match the game name, unless the content of the review is relevant to the question.
    If you don't know the answer, just say that you don't know. If the question is not related to the game, just say that you don't know.
    If the question is about a game not explicitly present in the context, just say that you don't know.
    Always verify that the game specified in the question is present in the context before answering.
    You are also provided with a chat history. If the question is related to the chat history, use it to answer the question. Always keep in mind the 
    chat history when answering the question.
    {chat_history}
    
    Context:
    {context}
    
    Question: {question}
    Answer:"""
    
    PROMPT = PromptTemplate(
        template=template,
        input_variables=["context", "question", "chat_history"]
    )
    
    # Set up LLM
    llm = ChatOpenAI(
        temperature=temperature, 
        model=model
    )
    
    # Class to customize document formatting for RAG with chat history support
    class CustomFormatRAG:
        def __init__(self, llm, retriever, prompt):
            self.llm = llm
            self.retriever = retriever
            self.prompt = prompt
        
        def invoke(self, params):
            query = params.get("query", "")
            chat_history = params.get("chat_history", [])
            previous_docs = params.get("previous_docs", None)
            reuse_context = params.get("reuse_context", False)
            
            # Format chat history if available
            chat_history_text = ""
            if chat_history:
                chat_history_text = "Chat History:\n" + "\n".join([
                    f"Human: {q}\nAI: {a}" for q, a in chat_history
                ])
            
            # Use previous docs if available and reuse_context is True, otherwise retrieve new ones
            if previous_docs and reuse_context:
                docs = previous_docs
            else:
                docs = self.retriever.get_relevant_documents(query)
            
            # Format documents to include game name
            formatted_context = format_documents(docs)
            
            # Execute the query with the LLM
            llm_response = self.llm.invoke(
                self.prompt.format(
                    context=formatted_context,
                    question=query,
                    chat_history=chat_history_text
                )
            )
            
            # Format the response to match the expected structure
            result = {
                "result": llm_response.content,
                "source_documents": docs
            }
            
            return result
    
    # Create and return our custom RAG implementation
    return CustomFormatRAG(llm, retriever, PROMPT)


# Create the RAG system with caching
@st.cache_resource
def load_chain(collection_name="steam_reviews", k=25, model="gpt-4o-mini", temperature=0):
    """Load and cache the RAG system."""
    return setup_rag_system(collection_name, k, model, temperature)


# Streamlit UI elements
def create_sidebar():
    """Create sidebar with configuration options."""
    st.sidebar.title("⚙️ Configuration")
    
    # Model selection
    model = st.sidebar.selectbox(
        "Select LLM Model",
        ["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4o", "gpt-4"],
        index=0
    )
    
    # Number of documents to retrieve
    k = st.sidebar.slider(
        "Number of reviews to retrieve (k)",
        min_value=5,
        max_value=50,
        value=25,
        step=5
    )
    
    # Temperature setting
    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1
    )
    
    # Show source documents option
    show_sources = st.sidebar.checkbox("Show source documents", value=False)
    
    # Add option to reuse context
    reuse_context = st.sidebar.checkbox(
        "Reuse previous context for follow-up questions", 
        value=True,
        help="When enabled, follow-up questions will use the same retrieved reviews instead of querying the database again."
    )
    
    # Add refresh context button
    refresh_context = st.sidebar.button(
        "Refresh Context", 
        help="Click to force retrieve new reviews for the next question even if context reuse is enabled."
    )
    
    return model, k, temperature, show_sources, reuse_context, refresh_context


# Main Streamlit app
def main():
    # Set up page configuration
    st.set_page_config(page_title="Steam Reviews RAG", layout="wide")
    st.title("🎮 Steam Reviews RAG Chatbot")
    st.markdown("Chat with Steam reviews to get insights about games")
    
    # Create sidebar and get configuration
    model, k, temperature, show_sources, reuse_context, refresh_context = create_sidebar()
    
    # Initialize chat memory in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "chat_memory" not in st.session_state:
        st.session_state.chat_memory = []
    
    if "retrieved_docs" not in st.session_state:
        st.session_state.retrieved_docs = None
    
    # Reset context if refresh button is clicked
    if refresh_context:
        st.session_state.retrieved_docs = None
        st.success("Context has been refreshed. The next question will retrieve new reviews.")
    
    # Load RAG chain with current configuration
    rag_system = load_chain(k=k, model=model, temperature=temperature)
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display source documents if enabled and available
            if show_sources and message["role"] == "assistant" and "sources" in message:
                with st.expander("View Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**Source {i}:** {source}")
    
    # Handle user input
    if prompt := st.chat_input("Ask about a game..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response with spinner
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Run query with chat history and previous docs if available
                response, metrics = run_rag_query(
                    rag_system, 
                    prompt, 
                    st.session_state.chat_memory,
                    st.session_state.retrieved_docs,
                    reuse_context
                )
                
                # Display context reuse notification if applicable
                if metrics.get("reused_context", False):
                    st.info("💡 Using previously retrieved context for this follow-up question.")
                
                # Display response
                result = response["result"]
                st.markdown(result)
                
                # Store retrieved documents for future questions
                st.session_state.retrieved_docs = response["source_documents"]
                
                # Store sources for display if requested
                sources = []
                if show_sources:
                    with st.expander("View Sources"):
                        for i, doc in enumerate(response["source_documents"], 1):
                            source_text = f"Review for \"{doc.metadata.get('game_name', 'Unknown Game')}\": {doc.page_content[:200]}..."
                            st.markdown(f"**Source {i}:** {source_text}")
                            sources.append(source_text)
        
        # Update chat history
        st.session_state.chat_memory.append((prompt, result))
        
        # Add assistant message to chat history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": result,
            "sources": sources if show_sources else []
        })


# Run the app
if __name__ == "__main__":
    # Check for LangSmith API key
    if not os.environ.get("LANGCHAIN_API_KEY"):
        st.warning("LANGCHAIN_API_KEY not set. LangSmith tracing will not work.")
    
    main()
