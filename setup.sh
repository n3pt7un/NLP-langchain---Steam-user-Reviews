#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored status messages
print_status() {
    echo -e "${YELLOW}[STATUS] $1${NC}"
}

# Function to print colored success messages
print_success() {
    echo -e "${GREEN}[SUCCESS] $1${NC}"
}

# Check if venv directory exists
if [ ! -d "venv" ]; then
    print_status "Creating Python virtual environment..."
    python -m venv venv
    print_success "Virtual environment created."
else
    print_status "Virtual environment already exists. Skipping creation."
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
print_status "Installing dependencies..."
pip install kagglehub>=0.3.11 pandas>=2.2.0 spacy>=3.8.0 contractions>=0.1.73 tqdm>=4.65.0 numpy>=1.26.0
pip install jupyter>=1.0.0 ipywidgets>=8.0.0 
pip install langsmith>=0.0.69 langchain>=0.1.0 langchain-openai>=0.0.3 langchain-qdrant>=0.0.1 langchain-community>=0.0.16
pip install streamlit>=1.32.0 qdrant-client>=1.6.0

# Check for requirements.txt and install if present
if [ -f "requirements.txt" ]; then
    print_status "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
fi

# Install spaCy English model
print_status "Installing spaCy English language model..."
python -m spacy download en_core_web_sm

# Verify spaCy model installation
print_status "Verifying spaCy model installation..."
python -c "import spacy; nlp = spacy.load('en_core_web_sm'); print('SpaCy model successfully loaded.')"

# Check if Qdrant is running locally (for convenience)
print_status "Checking if Qdrant is running locally..."
if command -v docker &> /dev/null; then
    if ! docker ps | grep -q "qdrant/qdrant"; then
        print_status "Qdrant not found. Consider running: docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant"
    else
        print_success "Qdrant is running locally."
    fi
else
    print_status "Docker not found. You'll need to run Qdrant separately."
    print_status "Install Docker and run: docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant"
fi

# Create an example .env file if it doesn't exist
if [ ! -f ".env" ]; then
    print_status "Creating example .env file..."
    echo 'OPENAI_API_KEY="your-openai-api-key"' > .env
    echo 'LANGCHAIN_TRACING_V2=true' >> .env
    echo 'LANGCHAIN_API_KEY="your-langchain-api-key"' >> .env
    echo 'LANGCHAIN_PROJECT="steam-reviews-rag"' >> .env
    print_status "Created .env file. Please update it with your actual API keys."
fi

print_success "Setup complete! The virtual environment is now activated and ready to use."
print_status "To deactivate the virtual environment, run 'deactivate'"
print_status "To run the Streamlit application: streamlit run main.py" 