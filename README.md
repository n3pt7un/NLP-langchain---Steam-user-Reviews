# Steam Reviews RAG Chatbot

## Overview
This project implements a Retrieval-Augmented Generation (RAG) system for Steam game reviews analysis. The application provides a chat-based interface that allows users to ask questions about games based on Steam user reviews, with responses generated using OpenAI's language models and relevant review context retrieved from a Qdrant vector database.

## Features
- Interactive chat interface with Streamlit frontend
- RAG system using Qdrant vector database and OpenAI embeddings
- Conversation memory to support follow-up questions
- Option to reuse context for follow-up questions (saves time and API calls)
- Configurable model selection (GPT-4o, GPT-4o-mini, GPT-3.5-turbo)
- Adjustable retrieval parameters for customizing the number of reviews used
- Source document display for transparency

## Project Structure
- `main.py`: The Streamlit application with chat interface and RAG implementation
- `rag_implementationV2.ipynb`: Initial notebook implementation of the RAG system
- `loading&preprocessing.ipynb`: Data loading and preprocessing notebook
- `games.csv`: List of Steam games for reference
- `setup.sh`: Script to set up the environment and install dependencies
- `requirements.txt`: List of Python package dependencies
- `.env.example`: Template for environment variables configuration

## Data Source
The data used in this project comes from the following Kaggle datasets:
1. [Steam Reviews Dataset](https://www.kaggle.com/datasets/filipkin/steam-reviews) by Filipkin
2. [Steam User Reviews (January Data)](https://www.kaggle.com/datasets/tarasivaniv/steam-user-reviews-preprocessed-january-data) by Taras Ivaniv

These datasets contain millions of user reviews across thousands of Steam games, which are processed and stored in a vector database for efficient retrieval.

## Data Preprocessing
The raw Steam review data undergoes several processing steps:
1. **Data Loading**: Reviews are loaded from Kaggle datasets and merged with game metadata
2. **Text Cleaning**: Reviews are cleaned by removing special characters, expanding contractions, etc.
3. **Tokenization & Lemmatization**: Using spaCy for linguistic processing
4. **Embedding Generation**: OpenAI's text-embedding models are used to create vector representations
5. **Vector Storage**: Processed and embedded reviews are stored in a Qdrant vector database

The detailed implementation of the preprocessing steps can be found in the `loading&preprocessing.ipynb` notebook.

## Setup and Installation

### Prerequisites
- Python 3.8 or higher
- Docker (for running Qdrant locally)
- OpenAI API key
- LangChain API key (optional, for tracing)

### Installation
1. Clone this repository
2. Run the setup script to install dependencies:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

3. Configure your API keys in the `.env` file:
   ```
   OPENAI_API_KEY="your-openai-api-key"
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_API_KEY="your-langchain-api-key"
   LANGCHAIN_PROJECT="steam-reviews-rag"
   ```

4. Start Qdrant using Docker:
   ```bash
   docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
   ```

### Data Preparation
To process and load the data into Qdrant:

1. Run the data loading and preprocessing notebook:
   ```bash
   jupyter notebook loading\&preprocessing.ipynb
   ```

2. Follow the steps in the notebook to download data from Kaggle, process it, and load it into Qdrant.

## Running the Application
Start the Streamlit application:
```bash
streamlit run main.py
```

The application will be available at http://localhost:8501 by default.

## Usage Guide

### Chat Interface
1. Enter your question about a game in the chat input field
2. The system will retrieve relevant reviews from the database
3. An AI response will be generated based on the retrieved reviews and your question
4. Chat history will be maintained for context in follow-up questions

### Configuration Options
Use the sidebar to customize the behavior:

- **Model Selection**: Choose between different OpenAI models
- **Number of reviews**: Adjust how many reviews to retrieve (k value)
- **Temperature**: Control response creativity (0 = more deterministic)
- **Show Sources**: Toggle to view the actual review sources
- **Reuse Context**: Enable to use the same context for follow-up questions
- **Refresh Context**: Force a new retrieval when needed

### Example Queries
- "What do players like about Elden Ring?"
- "What are common complaints about Counter-Strike: Global Offensive?"
- "How does the gameplay in The Witcher 3 compare to Skyrim?"
- "What features would players like to see improved in Stardew Valley?"

## Advanced Features

### Context Reuse for Follow-up Questions
This implementation includes a feature to reuse previously retrieved documents for follow-up questions. When enabled:

1. The first question retrieves documents from Qdrant as usual
2. Subsequent related questions reuse the same context
3. This saves API calls and time while maintaining conversation coherence
4. A notification appears when context is being reused
5. The "Refresh Context" button forces a new retrieval when needed

### LangSmith Tracing
For developers, LangSmith tracing is integrated to analyze and debug the RAG system:

1. Set your `LANGCHAIN_API_KEY` in the `.env` file
2. Set `LANGCHAIN_TRACING_V2=true` and a project name
3. View detailed traces of your queries in the LangSmith dashboard

## Troubleshooting
- **OpenAI API Error**: Ensure your API key is correctly set in the `.env` file
- **Qdrant Connection Error**: Make sure Qdrant is running on localhost:6333
- **No relevant reviews found**: Try a different game or rephrase your question
- **Follow-up questions not working**: Ensure the conversation is about the same topic

## Future Improvements
- Integration with real-time Steam review data
- Support for other vector databases
- Implementation of advanced retrieval techniques
- User authentication and personalized recommendations
- Multi-modal support for game screenshots and videos

## License
This project is released under the MIT License.

## Acknowledgments
- OpenAI for language models and embeddings
- Qdrant for the vector database
- LangChain for RAG components
- Streamlit for the chat interface
- Kaggle dataset contributors for providing Steam review data 