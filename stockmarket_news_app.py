# Import required libraries
import os
import re
from dotenv import load_dotenv
import asyncio
from langchain_community.document_loaders import WebBaseLoader

# Load environment variables
load_dotenv('./../.env')

# URLs to scrape for stock market data
STOCK_MARKET_URLS = [
    'https://economictimes.indiatimes.com/markets/stocks/news',
    'https://www.livemint.com/latest-news',
    'https://www.livemint.com/latest-news/page-2',
    'https://www.livemint.com/latest-news/page-3',
    'https://www.moneycontrol.com/'
]


def clean_text(text):
    """
    Clean the extracted text by removing excessive whitespace and formatting

    Args:
        text: Raw text from webpage

    Returns:
        Cleaned text with normalized whitespace
    """
    text = re.sub(r'\n\n+', '\n\n', text)
    text = re.sub(r'\t+', '\t', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def chunk_text(text, chunk_size, overlap=100):
    """
    Break text into overlapping chunks to process with LLM

    Args:
        text: Text to be chunked
        chunk_size: Maximum size of each chunk
        overlap: Number of characters to overlap between chunks

    Returns:
        List of text chunks
    """
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks


def format_docs(docs):
    """
    Format documents into a single string

    Args:
        docs: List of document objects

    Returns:
        String containing all document content
    """
    return "\n\n".join([x.page_content for x in docs])


async def load_webpages(urls):
    """
    Load and process content from specified URLs

    Args:
        urls: List of URLs to scrape

    Returns:
        List of document objects containing webpage content
    """
    loader = WebBaseLoader(web_paths=urls)
    docs = []
    async for doc in loader.alazy_load():
        docs.append(doc)
    return docs


def process_with_llm(text, question, llm_module):
    """
    Process text with LLM to get answers to questions

    Args:
        text: Text to be analyzed
        question: Question to ask the LLM
        llm_module: Module containing the ask_llm function

    Returns:
        LLM response to the question
    """
    return llm_module.ask_llm(text, question)


def save_to_file(content, filename, directory="data"):
    """
    Save content to a file

    Args:
        content: Content to save
        filename: Name of the file
        directory: Directory to save the file in
    """
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, filename), "w") as f:
        f.write(content)


async def main():
    """Main function to run the webpage processing pipeline"""

    # Import LLM module (assuming it exists in scripts directory)
    from scripts import llm

    # Load and process webpages
    print("Loading webpages...")
    docs = await load_webpages(STOCK_MARKET_URLS)
    print(f"Loaded {len(docs)} webpages")

    # Format and clean the text
    context = format_docs(docs)
    context = clean_text(context)

    # Process text in chunks due to context length limitations
    print("Processing text in chunks...")
    chunks = chunk_text(context, 10_000)

    # Ask LLM to extract stock market news from each chunk
    question = "Extract stock market news from the given text."
    chunk_summaries = []

    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i + 1}/{len(chunks)}...")
        response = process_with_llm(chunk, question, llm)
        chunk_summaries.append(response)

    # Combine all chunk summaries
    summary = "\n\n".join(chunk_summaries)

    # Generate final report in Markdown format
    print("Generating final report...")
    report_question = "Write a detailed market news report in markdown format. Think carefully then write the report."
    final_report = process_with_llm(summary, report_question, llm)

    # Save results
    save_to_file(summary, "summary.md")
    save_to_file(final_report, "report.md")

    print("Processing complete. Files saved in the data directory.")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())