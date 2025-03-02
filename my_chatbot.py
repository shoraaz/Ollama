
# Import required libraries
import streamlit as st  # For building the web interface
from dotenv import load_dotenv  # For loading environment variables
from langchain_ollama import ChatOllama  # For connecting to Ollama LLM

# Import components for creating chat prompts
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)

# Import components for managing chat history
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory

# Import parser to convert LLM output to string
from langchain_core.output_parsers import StrOutputParser

# Load environment variables from .env file
load_dotenv('./../.env')

# Set up the Streamlit UI
st.title("My chatbot")

# Define the LLM model to use
model = 'llama3.2:3b'

# Get user ID for maintaining chat history
user_id = st.text_input("Enter your user id", "shoraaz")

# Function to retrieve chat history from SQLite database
def get_session_history(session_id):
    return SQLChatMessageHistory(session_id, "sqlite:///chat_history.db")

# Initialize chat history in session state if it doesn't exist
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Button to start a new conversation
if st.button("Start New Conversation"):
    st.session_state.chat_history = []
    history = get_session_history(user_id)
    history.clear()

# Display previous chat messages
for message in st.session_state.chat_history:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# Set up the LLM chain
# Initialize the LLM with the selected model
llm = ChatOllama(model=model)

# Define message templates
system = SystemMessagePromptTemplate.from_template("You are helpful assistant.")
human = HumanMessagePromptTemplate.from_template("{input}")

# Create a prompt template with system message, message history, and human input
messages = [system, MessagesPlaceholder(variable_name='history'), human]
prompt = ChatPromptTemplate(messages=messages)

# Create the processing chain: prompt -> LLM -> string output
chain = prompt | llm | StrOutputParser()

# Add history management to the chain
runnable_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key='input',
    history_messages_key='history'
)

# Function to stream chat responses
def chat_with_llm(session_id, input):
    for output in runnable_with_history.stream(
        {'input': input},
        config={'configurable': {'session_id': session_id}}
    ):
        yield output

# Chat input field
prompt = st.chat_input("What is up?")

# Process user input when submitted
if prompt:
    # Add user message to chat history
    st.session_state.chat_history.append({'role': 'user', 'content': prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant's response with streaming
    with st.chat_message("assistant"):
        response = st.write_stream(chat_with_llm(user_id, prompt))

    # Add assistant's response to chat history
    st.session_state.chat_history.append({'role': 'assistant', 'content': response})