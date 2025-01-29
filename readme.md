# Multi-Document Journal Summarization

A Streamlit application that summarizes multiple academic journal PDFs using both extractive and abstractive summarization techniques. This tool helps researchers and students quickly grasp the key points from multiple academic papers.

## Features

- Upload multiple PDF documents simultaneously
- Extract text content from PDF files
- Preprocess text to remove stopwords and noise
- Two summarization modes:
  - Extractive summarization using LSA (Latent Semantic Analysis)
  - Abstractive summarization using BART (facebook/bart-large-cnn)
- Interactive web interface built with Streamlit

## Installation

1. Clone the repository:
```bash
git clone <https://github.com/EngRidhoNet/summarization_streamlit>
cd multi-document-summarizer
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Download NLTK resources:
```bash
python -m nltk.downloader stopwords
python -m nltk.downloader punkt
```

## Dependencies

- Python 3.7+
- NLTK
- Sumy
- Transformers
- Streamlit
- PyPDF2
- torch

Create a requirements.txt file with the following:
```
nltk
sumy
transformers
streamlit
PyPDF2
torch
```

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Access the web interface through your browser (typically http://localhost:8501)

3. Upload your PDF files using the file uploader

4. Select your preferred summarization mode:
   - Extractive: Uses LSA to extract key sentences from the text
   - Abstractive: Uses BART to generate a new summary

5. Click "Summarize" to process the documents

## Application Structure

```
├── app.py                 # Main application file
├── requirements.txt       # Project dependencies
└── README.md             # Documentation
```

## Functions Description

- `extract_text_from_pdf(pdf_file)`: Extracts text content from uploaded PDF files
- `preprocess_text(text)`: Removes stopwords and performs basic text cleaning
- `extractive_summarization(text, ratio)`: Performs LSA-based extractive summarization
- `abstractive_summarization(text, max_length, min_length)`: Performs BART-based abstractive summarization
- `multi_document_summarization(journals, mode)`: Handles multiple document summarization

## Configuration

The application uses several configurable parameters:

- Extractive summarization ratio: 0.2 (20% of original text)
- Abstractive summarization:
  - Maximum length: 130 tokens
  - Minimum length: 30 tokens
- NLTK data path: Customize in the code according to your environment

## Troubleshooting

Common issues and solutions:

1. NLTK Resource Error:
   - Ensure NLTK resources are properly downloaded
   - Check NLTK data path configuration

2. PDF Extraction Issues:
   - Verify PDF is not encrypted
   - Check PDF file permissions
   - Ensure PDF contains extractable text

3. Memory Issues:
   - Reduce batch size of documents
   - Process larger documents individually

## Limitations

- Maximum token limit for abstractive summarization: 1024 tokens
- PDF must contain extractable text (non-scanned documents)
- Processing time increases with document length and quantity

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

