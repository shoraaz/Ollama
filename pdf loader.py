# Import required libraries
import os
from dotenv import load_dotenv
import tiktoken
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_ollama import ChatOllama
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate
)
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv('./../.env')

# Define Ollama configuration
base_url = "http://localhost:11434"
model = 'llama3.2:3b'


def find_pdf_files(directory_path="rag-dataset"):
    """
    Find all PDF files in the specified directory and subdirectories

    Args:
        directory_path: Root directory to search for PDF files

    Returns:
        List of paths to PDF files
    """
    pdfs = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".pdf"):
                pdfs.append(os.path.join(root, file))
    return pdfs


def load_documents(pdf_paths):
    """
    Load documents from PDF files

    Args:
        pdf_paths: List of paths to PDF files

    Returns:
        List of document objects
    """
    docs = []
    for pdf in pdf_paths:
        loader = PyMuPDFLoader(pdf)
        temp = loader.load()
        docs.extend(temp)
    return docs


def format_docs(docs):
    """
    Format documents into a single string

    Args:
        docs: List of document objects

    Returns:
        String containing all document content
    """
    return "\n\n".join([x.page_content for x in docs])


def count_tokens(text):
    """
    Count tokens in text using tiktoken

    Args:
        text: String to count tokens for

    Returns:
        Number of tokens
    """
    encoding = tiktoken.encoding_for_model("gpt-4o-mini")
    return len(encoding.encode(text))


def create_qa_chain():
    """
    Create a question answering chain

    Returns:
        QA chain for answering questions based on context
    """
    # Initialize the LLM
    llm = ChatOllama(base_url=base_url, model=model)

    # Create system message template
    system = SystemMessagePromptTemplate.from_template(
        "You are helpful AI assistant who answer user question based on the provided context. "
        "Do not answer in more than {words} words"
    )

    # Create human message template
    prompt = """Answer user question based on the provided context ONLY! If you do not know the answer, just say "I don't know".
                ### Context:
                {context}

                ### Question:
                {question}

                ### Answer:"""
    prompt = HumanMessagePromptTemplate.from_template(prompt)

    # Create chat template with system and human messages
    messages = [system, prompt]
    template = ChatPromptTemplate(messages)

    # Create and return the chain
    return template | llm | StrOutputParser()


def create_summary_chain():
    """
    Create a document summarization chain

    Returns:
        Chain for summarizing document content
    """
    # Initialize the LLM
    llm = ChatOllama(base_url=base_url, model=model)

    # Create system message template
    system = SystemMessagePromptTemplate.from_template(
        "You are helpful AI assistant who works as document summarizer. "
        "You must not hallucinate or provide any false information."
    )

    # Create human message template
    prompt = """Summarize the given context in {words}.
                ### Context:
                {context}

                ### Summary:"""
    prompt = HumanMessagePromptTemplate.from_template(prompt)

    # Create chat template with system and human messages
    messages = [system, prompt]
    template = ChatPromptTemplate(messages)

    # Create and return the chain
    return template | llm | StrOutputParser()


def main():
    """Main function to demonstrate PDF document processing capabilities"""

    # Find all PDF files in the dataset directory
    pdf_paths = find_pdf_files()
    print(f"Found {len(pdf_paths)} PDF files")

    # Load documents from PDF files
    docs = load_documents(pdf_paths)
    print(f"Loaded {len(docs)} document chunks")

    # Format documents into a single context
    context = format_docs(docs)
    token_count = count_tokens(context)
    print(f"Total context size: {token_count} tokens")

    # Create QA and summary chains
    qa_chain = create_qa_chain()
    summary_chain = create_summary_chain()

    # Example: Question answering
    question = "How to gain muscle mass?"
    print(f"\nQuestion: {question}")
    answer = qa_chain.invoke({'context': context, 'question': question, 'words': 50})
    print(f"Answer: {answer}")

    # Example: Document summarization
    print("\nGenerating short summary (50 words)...")
    short_summary = summary_chain.invoke({'context': context, 'words': 50})
    print(f"Short summary: {short_summary}")

    print("\nGenerating longer summary (500 words)...")
    long_summary = summary_chain.invoke({'context': context, 'words': 500})
    print(f"Longer summary: {long_summary}")


if __name__ == "__main__":
    main()