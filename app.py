import logging
from transformers import AutoTokenizer
import os
import nltk
from nltk.corpus import stopwords
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
from transformers import pipeline
import streamlit as st
from PyPDF2 import PdfReader
from transformers import pipeline

# Sesuaikan dengan path yang sesuai
nltk.data.path.append("C:/Users/dhima/nltk_data")
nltk.download('stopwords')
nltk.download('punkt_tab')


def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    words = nltk.word_tokenize(text)
    filtered_words = [word for word in words if word.lower(
    ) not in stop_words and word.isalnum()]
    return " ".join(filtered_words)


def extractive_summarization(text, ratio=0.2):
    try:
        # Initialize the summarizer
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        stemmer = Stemmer("english")
        summarizer = LsaSummarizer(stemmer)
        summarizer.stop_words = get_stop_words("english")

        # Calculate the number of sentences based on ratio
        sentence_count = max(int(len(parser.document.sentences) * ratio), 1)

        # Get the summary
        summary = [str(sentence) for sentence in summarizer(
            parser.document, sentence_count)]
        return " ".join(summary)
    except Exception as e:
        return f"Error in summarization: {str(e)}"
    

# def truncate_text(text, max_tokens=1024):
#     words = text.split()
#     return " ".join(words[:max_tokens])  # Ambil maksimal 1024 token


summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


# def abstractive_summarization(text, max_length=130, min_length=30):
#     truncated_text = truncate_text(text, max_tokens=1024)  # Batasi input
#     try:
#         summary = summarizer(truncated_text, max_length=max_length,
#                              min_length=min_length, do_sample=False)
#         return summary[0]['summary_text']
#     except Exception as e:
#         return f"Error: {str(e)}"




# def abstractive_summarization(text, max_length=130, min_length=30):
#     summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
#     summary = summarizer(text, max_length=max_length,
#                          min_length=min_length, do_sample=False)
#     return summary[0]['summary_text']
# def multi_document_summarization(journals, mode="extractive", ratio=0.1, max_length=150, min_length=50):
#     combined_text = " ".join(journals)
#     # Pastikan teks tidak terlalu panjang
#     truncated_text = truncate_text(combined_text, max_tokens=1024)
#     if mode == "extractive":
#         return extractive_summarization(truncated_text, ratio=ratio)
#     elif mode == "abstractive":
#         return abstractive_summarization(truncated_text, max_length=max_length, min_length=min_length)
#     else:
#         return "Invalid mode. Choose 'extractive' or 'abstractive'."
# Set up logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize tokenizer globally along with the summarizer
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")


def safe_summarize(text, max_length=130, min_length=30):
    """Helper function to safely run the summarization pipeline"""
    try:
        # Log the input text length
        logger.info(f"Input text length: {len(text)} characters")

        # Ensure the text isn't empty and is substantial enough
        if not text.strip() or len(text.split()) < 10:  # At least 10 words
            logger.warning("Text too short for summarization")
            return None

        # Get token count
        tokens = tokenizer.encode(text, truncation=False, return_tensors="pt")
        logger.info(f"Token count: {tokens.shape[1]}")

        if tokens.shape[1] > 1024:
            logger.info("Truncating text to 1024 tokens")
            text = tokenizer.decode(tokens[0][:1024], skip_special_tokens=True)

        # Attempt summarization
        summary = summarizer(text,
                             max_length=max_length,
                             min_length=min_length,
                             do_sample=False,
                             truncation=True)

        if summary and summary[0]['summary_text']:
            logger.info("Successfully generated summary")
            return summary[0]['summary_text']
        else:
            logger.warning("Summarizer returned empty result")
            return None

    except Exception as e:
        logger.error(f"Error in summarization: {str(e)}")
        return None


def abstractive_summarization(text, max_length=130, min_length=30):
    if not text:
        logger.error("No text provided for summarization")
        return "No text provided for summarization."

    try:
        # Create more manageable chunks
        words = text.split()
        chunk_size = 512  # Smaller chunks to ensure processing
        chunks = []

        # Create chunks of approximately 512 words
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():  # Only add non-empty chunks
                chunks.append(chunk)

        logger.info(f"Created {len(chunks)} chunks")

        if not chunks:
            logger.error("No valid chunks created")
            return "Text processing failed - no valid chunks created."

        # Process each chunk
        summaries = []
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}")
            chunk_summary = safe_summarize(chunk,
                                           max_length=max_length,
                                           min_length=min(min_length, len(chunk.split())//2))
            if chunk_summary:
                summaries.append(chunk_summary)
            logger.info(
                f"Chunk {i+1} {'successfully' if chunk_summary else 'failed to'} summarize")

        # If we got no summaries
        if not summaries:
            logger.error("No valid summaries generated from any chunks")
            return "Could not generate any valid summaries. Please check if the input text is valid and contains enough content to summarize."

        # Combine summaries
        final_summary = ' '.join(summaries)
        logger.info(f"Combined summary length: {
                    len(final_summary)} characters")

        # If combined summary is too long, summarize it again
        if len(tokenizer.encode(final_summary)) > max_length * 2:
            logger.info("Performing final summarization pass")
            final_summary = safe_summarize(
                final_summary, max_length=max_length, min_length=min_length)
            if not final_summary:
                return "Failed to generate final summary."

        return final_summary
    except Exception as e:
        logger.error(f"Error in abstractive summarization: {str(e)}")
        return f"Error in summarization: {str(e)}"


def multi_document_summarization(journals, mode="extractive", ratio=0.1, max_length=150, min_length=50):
    if not journals:
        return "No documents provided for summarization."

    try:
        logger.info(f"Processing {len(journals)} documents in {mode} mode")
        combined_text = " ".join(journals)
        logger.info(f"Combined text length: {len(combined_text)} characters")

        if mode.lower() == "extractive":
            return extractive_summarization(combined_text, ratio=ratio)
        elif mode.lower() == "abstractive":
            return abstractive_summarization(combined_text, max_length=max_length, min_length=min_length)
        else:
            return "Invalid mode. Choose 'extractive' or 'abstractive'."
    except Exception as e:
        logger.error(f"Error in multi-document summarization: {str(e)}")
        return f"Error in multi-document summarization: {str(e)}"


def main():
    st.title("Multi-Document Journal Summarization")
    st.write("Upload multiple journal PDFs and get a summary!")

    # Upload PDF files
    uploaded_files = st.file_uploader(
        "Upload PDF files", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        journals = []
        for uploaded_file in uploaded_files:
            text = extract_text_from_pdf(uploaded_file)
            preprocessed_text = preprocess_text(text)
            journals.append(preprocessed_text)

        st.write(f"Loaded {len(journals)} journals.")

        # Summarization mode selection
        mode = st.radio("Select Summarization Mode",
                        ("Extractive", "Abstractive"))

        if st.button("Summarize"):
            with st.spinner("Generating summary..."):
                if mode == "Extractive":
                    summary = multi_document_summarization(
                        journals, mode="extractive")
                else:
                    summary = multi_document_summarization(
                        journals, mode="abstractive")

                st.subheader("Summary:")
                st.write(summary)


if __name__ == "__main__":
    main()
