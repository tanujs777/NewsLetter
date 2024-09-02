import validators
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Set up Google Gemini API key and Hugging Face token
gemini_api_key = os.getenv("GEMINI_API_KEY")
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN", "your-hf-token")

# Set up Streamlit page configuration
st.set_page_config(page_title="Product Page Summarizer")
st.title("Product Page Summarizer")
st.subheader("Summarize the key details of a product page")

# Sidebar input for Web URL
with st.sidebar:
    web_url = st.text_input("Enter Product Page URL")

# Initialize the Google Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0.7,
    max_tokens=200,
    timeout=30,
    max_retries=2,
    api_key=gemini_api_key  # Pass the API key directly
)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Define the system template for summarization
system_template = (
    "You are a product description summarizer. Your task is to read the product page content "
    "and provide a concise summary that includes the following key details:\n"
    "1. Product Name\n"
    "2. Key Features\n"
    "3. Price\n"
    "4. Customer Reviews\n\n"
    "{context}"
)

# User input for chat query
chat_query = "Please summarize the product page."

# Handle the summarization when the user clicks the "Summarize" button
if st.button("Summarize"):
    try:
        # Input validation
        if not web_url:
            st.error("Please provide the Product Page URL to get started.")
        elif not validators.url(web_url):
            st.error("Please enter a valid URL.")
        else:
            # Load and process the website content
            loader = WebBaseLoader(web_path=[web_url])
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(separators="\n", chunk_size=1000, chunk_overlap=200)
            final_docs = text_splitter.split_documents(docs)
            
            # Create a FAISS vector store from the documents
            vector_store_db = FAISS.from_documents(final_docs, embeddings)
            retriever = vector_store_db.as_retriever()
            
            # Create a prompt and the summarization chain
            prompt = PromptTemplate(input_variables=["context"], template=system_template)
            summary_chain = create_stuff_documents_chain(llm, prompt)
            rag_chain = create_retrieval_chain(retriever, summary_chain)
            
            # Get the summary from the chain
            response = rag_chain.invoke({"input": chat_query})
            st.success("Summary Generated:")
            st.json(response["answer"])
    except Exception as e:
        st.exception(f'Exception: {e}')
