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
print_status "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Install spaCy English model
print_status "Installing spaCy English language model..."
python -m spacy download en_core_web_sm

# Verify spaCy model installation
print_status "Verifying spaCy model installation..."
python -c "import spacy; nlp = spacy.load('en_core_web_sm'); print('SpaCy model successfully loaded.')"

print_success "Setup complete! The virtual environment is now activated and ready to use."
print_status "To deactivate the virtual environment, run 'deactivate'" 