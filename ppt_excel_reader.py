# Import required libraries
import os
import nltk
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
    Docx2txtLoader
)

# Load environment variables
load_dotenv('./../.env')


# Ensure NLTK punkt tokenizer is downloaded
def setup_nltk():
    """
    Setup NLTK by downloading required data
    """
    nltk.download('punkt', quiet=True)


def load_ppt(file_path="data/ml_course.pptx"):
    """
    Load and process PowerPoint presentation file

    Args:
        file_path: Path to the PowerPoint file

    Returns:
        Dictionary of slide content by page number and concatenated context
    """
    # Load PowerPoint using Unstructured
    loader = UnstructuredPowerPointLoader(file_path, mode="elements")
    docs = loader.load()

    # Organize content by slide number
    ppt_data = {}
    for doc in docs:
        page = doc.metadata["page_number"]
        ppt_data[page] = ppt_data.get(page, "") + "\n\n" + doc.page_content

    # Format context with slide numbers
    context = ""
    for page, content in ppt_data.items():
        context += f"### Slide {page}:\n\n{content.strip()}\n\n\n"

    return ppt_data, context


def generate_ppt_script(context, llm_module):
    """
    Generate speaker script for PowerPoint slides

    Args:
        context: Formatted slide content
        llm_module: Module containing the ask_llm function

    Returns:
        Generated script for the presentation
    """
    question = """
    For each PowerPoint slide provided above, write a 2-minute script that effectively conveys the key points.
    Ensure a smooth flow between slides, maintaining a clear and engaging narrative.
    """

    response = llm_module.ask_llm(context, question)
    return response


def load_excel(file_path="data/sample.xlsx"):
    """
    Load and process Excel file

    Args:
        file_path: Path to the Excel file

    Returns:
        HTML representation of the Excel data
    """
    loader = UnstructuredExcelLoader(file_path, mode="elements")
    docs = loader.load()

    # Get HTML representation of the table
    context = docs[0].metadata['text_as_html']
    return context


def analyze_excel_data(context, query, llm_module):
    """
    Ask questions about Excel data using LLM

    Args:
        context: HTML representation of Excel data
        query: Question to ask about the data
        llm_module: Module containing the ask_llm function

    Returns:
        LLM response to the query
    """
    response = llm_module.ask_llm(context, query)
    return response


def load_word_document(file_path="data/job_description.docx"):
    """
    Load and process Word document

    Args:
        file_path: Path to the Word document

    Returns:
        Text content of the Word document
    """
    loader = Docx2txtLoader(file_path)
    docs = loader.load()
    context = docs[0].page_content
    return context


def generate_job_application(context, applicant_info, llm_module):
    """
    Generate job application based on job description and applicant info

    Args:
        context: Job description text
        applicant_info: Information about the applicant
        llm_module: Module containing the ask_llm function

    Returns:
        Generated job application letter
    """
    question = f"{applicant_info}\nPlease write a concise job application email for me in short, removing any placeholders."
    response = llm_module.ask_llm(context, question)
    return response


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


def main():
    """Main function demonstrating document processing capabilities"""

    # Import LLM module
    from scripts import llm

    setup_nltk()
    print("NLTK setup complete")

    # Example 1: PowerPoint Processing
    print("\nProcessing PowerPoint presentation...")
    ppt_data, ppt_context = load_ppt()
    script = generate_ppt_script(ppt_context, llm)
    save_to_file(script, "ppt_script.md")
    print("PowerPoint script generated and saved")

    # Example 2: Excel Data Analysis
    print("\nProcessing Excel data...")
    excel_context = load_excel()

    # Analyze females in the dataset
    females_query = "Return all entries in the table where Gender is 'F'. Format the response in Markdown. Do not write preambles and explanation."
    females_analysis = analyze_excel_data(excel_context, females_query, llm)
    save_to_file(females_analysis, "females_analysis.md")

    # Analyze males in the dataset
    males_query = "Return all entries in the table where Gender is 'male'. Format the response in Markdown. Do not write preambles and explanation."
    males_analysis = analyze_excel_data(excel_context, males_query, llm)
    save_to_file(males_analysis, "males_analysis.md")
    print("Excel analysis complete and results saved")

    # Example 3: Word Document Processing
    print("\nProcessing Word document...")
    job_description = load_word_document()

    applicant_info = """
    My name is Aaditya, and I am a recent graduate from IIT with a focus on Natural Language Processing and Machine Learning.
    I am applying for a Data Scientist position at SpiceJet.
    """

    application = generate_job_application(job_description, applicant_info, llm)
    save_to_file(application, "job_application.md")
    print("Job application generated and saved")

    print("\nAll processing complete. Files saved in the data directory.")


if __name__ == "__main__":
    main()