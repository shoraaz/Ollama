import os
import warnings
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

# Suppress warnings and environment setup
warnings.filterwarnings("ignore")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
load_dotenv()


def load_vector_store(db_path):
    """Load the FAISS vector store from a local directory"""
    embeddings = OllamaEmbeddings(model='nomic-embed-text', base_url='http://localhost:11434')
    return FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)


def format_docs(docs):
    """Format the retrieved documents into a single string"""
    return '\n\n'.join([doc.page_content for doc in docs])


def create_rag_chain(vector_store):
    """Create a RAG chain combining retrieval and generation"""
    # Create a retriever with MMR search strategy
    retriever = vector_store.as_retriever(
        search_type='mmr',
        search_kwargs={'k': 3, 'fetch_k': 20, 'lambda_mult': 1}
    )

    # Define the prompt template
    prompt_template = """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    Answer in bullet points. Make sure your answer is relevant to the question and it is answered from the context only.

    Question: {question}
    Context: {context}
    Answer:
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)

    # Initialize the LLM
    llm = ChatOllama(model='llama3.2:3b', base_url='http://localhost:11434')

    # Create the RAG chain
    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    return rag_chain


def answer_question(rag_chain, question):
    """Get an answer to a question using the RAG chain"""
    response = rag_chain.invoke(question)
    return response


def main():
    # Path to the vector store
    db_path = "health_supplements"  # Update this path as needed

    print("Loading vector store...")
    vector_store = load_vector_store(db_path)

    print("Creating RAG chain...")
    rag_chain = create_rag_chain(vector_store)

    # Example usage
    questions = [
        "how to lose weight?",
        "how to gain muscle mass?"
    ]

    for question in questions:
        print(f"\nQuestion: {question}")
        answer = answer_question(rag_chain, question)
        print("Answer:")
        print(answer)


if __name__ == "__main__":
    main()