# Steam Reviews NLP Pipeline

## Overview
This project provides a natural language processing (NLP) pipeline for analyzing Steam game reviews. It handles the complete workflow from data loading and preprocessing to embedding generation and storage for retrieval-augmented generation (RAG) applications.

## Features
- Data acquisition from Kaggle datasets of Steam reviews
- Comprehensive text preprocessing:
  - Basic cleanup (lowercase, contraction expansion, special character removal)
  - Tokenization, stopword removal, and lemmatization using spaCy
  - Filtering based on token frequency and document frequency thresholds
- Efficient processing with multiprocessing and batch processing
- Vector storage integration (partial implementation)
- Support for RAG applications (in progress)

## Project Structure
- `main.ipynb`: Jupyter notebook containing the full pipeline implementation
- `setup.sh`: Script to set up the virtual environment and install dependencies
- `requirements.txt`: List of Python package dependencies
- `.env`: Environment variables file for API keys

## Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Access to Kaggle API (for dataset download)

### Installation
1. Clone this repository
2. Run the setup script:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```
   This will:
   - Create a virtual environment
   - Install all required packages
   - Download the spaCy English language model

3. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```

4. (Optional) Configure OpenAI API key in the `.env` file if you plan to use OpenAI embeddings.

## Usage
1. Ensure your virtual environment is activated
2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
3. Open `main.ipynb`
4. Run the cells sequentially to:
   - Load data from Kaggle
   - Process and clean the text
   - Generate embeddings (requires additional setup)
   - Store in vector database (requires additional setup)

## Data Processing Pipeline
The processing pipeline consists of these main steps:

1. **Data Loading**: Uses kagglehub to download datasets of Steam reviews and game metadata, then merges them.
2. **Basic Text Cleaning**: 
   - Converts text to lowercase
   - Expands contractions (e.g., "don't" → "do not")
   - Removes special characters
3. **Advanced NLP Processing**:
   - Tokenization using spaCy
   - Removal of stopwords, punctuation, and numbers
   - Lemmatization to normalize word forms
4. **Token Filtering**:
   - Removes rare words (appearing less than a threshold)
   - Filters based on document frequency (too common or too rare)
5. **Embedding and Storage** (partial implementation):
   - Generates embeddings for processed text
   - Stores vectors in Qdrant for retrieval

## Dependencies
- kagglehub: For dataset access
- pandas: For data manipulation
- spaCy: For NLP processing
- contractions: For expanding contractions
- tqdm: For progress tracking
- numpy: For numerical operations
- Jupyter: For notebook interface

**Additional dependencies for embedding/storage** (not fully implemented):
- langchain-community
- sentence-transformers
- qdrant-client
- OpenAI API

## Enhanced Preprocessing Function Example

Below is a comprehensive function that demonstrates text preprocessing using a more compact approach:

```python
import pandas as pd
import spacy
import contractions
import re
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

def preprocess_data(df, text_column='content', rare_threshold=5, min_doc_freq=2, max_doc_freq_ratio=0.8):
    """
    Preprocesses the input dataframe containing text data.
    
    Steps:
      1. Remove missing values and duplicates.
      2. Lowercase text, fix contractions, and remove special characters.
      3. Use spaCy to tokenize, remove stopwords and punctuation, and lemmatize tokens.
      4. Remove rare words (appearing less than rare_threshold times across the corpus).
      5. Remove words that appear in less than min_doc_freq documents or in more than 
         max_doc_freq_ratio * (number of documents) of the corpus.
    
    Parameters:
      df (pd.DataFrame): DataFrame containing the text data.
      text_column (str): Name of the column containing raw text.
      rare_threshold (int): Minimum frequency for a token to be kept.
      min_doc_freq (int): Minimum number of documents a token must appear in.
      max_doc_freq_ratio (float): Maximum proportion of documents in which a token may appear.
      
    Returns:
      pd.DataFrame: DataFrame with additional columns for processed text.
    """
    # 1. Clean the dataframe
    df = df.dropna().drop_duplicates().reset_index(drop=True)
    
    # 2. Basic cleaning: lowercase, fix contractions, remove special characters
    def basic_clean(text):
        text = text.lower()
        text = contractions.fix(text)
        # Remove any character that is not alphanumeric or whitespace
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return text
    
    df['cleaned'] = df[text_column].apply(basic_clean)
    
    # 3. Process text with spaCy for tokenization, stopword removal, and lemmatization
    nlp = spacy.load("en_core_web_sm")
    
    def spacy_process(text):
        doc = nlp(text)
        # Keep the lemma of tokens that are not stopwords, punctuation, or numbers
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and not token.like_num]
        return tokens
    
    df['tokens'] = df['cleaned'].apply(spacy_process)
    
    # 4. Compute overall token frequency (for rare word removal)
    all_tokens = [token for tokens in df['tokens'] for token in tokens]
    token_freq = Counter(all_tokens)
    rare_words = {word for word, freq in token_freq.items() if freq < rare_threshold}
    
    # 5. Compute document frequency filtering using CountVectorizer
    # Convert token lists back to strings for vectorization
    df['joined_tokens'] = df['tokens'].apply(lambda tokens: " ".join(tokens))
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['joined_tokens'])
    doc_freq_array = X.toarray().sum(axis=0)
    doc_freq = dict(zip(vectorizer.get_feature_names_out(), doc_freq_array))
    num_docs = len(df)
    # Filter words that appear in an acceptable number of documents
    filtered_words = {word for word, freq in doc_freq.items() 
                      if freq >= min_doc_freq and freq <= max_doc_freq_ratio * num_docs}
    
    # 6. Filter tokens: remove tokens that are rare or not within the acceptable document frequency range
    def filter_tokens(tokens):
        return [token for token in tokens if token not in rare_words and token in filtered_words]
    
    df['final_tokens'] = df['tokens'].apply(filter_tokens)
    # Optionally, join tokens back into a single string
    df['final_text'] = df['final_tokens'].apply(lambda tokens: " ".join(tokens))
    
    return df
```

This function provides a more compact implementation compared to the current project implementation but achieves similar functionality. It could be used as a reference for future optimizations.

## Current Limitations and Future Work
- Vector database integration is currently incomplete due to package compatibility issues
- LangChain integration needs additional development
- No final RAG application interface has been implemented
- Additional filtering and preprocessing options can be added
- Performance optimizations for larger datasets

## License
MIT License

## Acknowledgments
- Kaggle for providing the datasets
- Contributors to the open-source libraries used in this project

---

*This README was generated with AI assistance* 