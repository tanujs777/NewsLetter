import validators
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv
import os


load_dotenv()


groq_api_key = os.getenv("GROQ_API_KEY")
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN", "your-hf-token")


st.set_page_config(page_title="Product Page Summarizer")
st.title("Product Page Summarizer")
st.subheader("Summarize the key details of a product page")

with st.sidebar:
    web_url = st.text_input("Enter Product Page URL")

llm = ChatGroq(groq_api_key=groq_api_key, model="Gemma2-9b-It")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

system_template = (
    "You are a product description summarizer. Your task is to read the product page content "
    "and provide a concise summary that includes the following key details:\n"
    "1. Product Name\n"
    "2. Key Features\n"
    "3. Price\n"
    "4. Customer Reviews\n\n"
    "{context}"
)

chat_query = "Please summarize the product page."


if st.button("Summarize"):
    try:
        if not web_url:
            st.error("Please provide the Product Page URL to get started.")
        elif not validators.url(web_url):
            st.error("Please enter a valid URL.")
        else:
            loader = WebBaseLoader(web_path=[web_url])
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(separators="\n", chunk_size=1000, chunk_overlap=200)
            final_docs = text_splitter.split_documents(docs)
            
            vector_store_db = FAISS.from_documents(final_docs, embeddings)
            retriever = vector_store_db.as_retriever()
            
            prompt = PromptTemplate(input_variables=["context"], template=system_template)
            summary_chain = create_stuff_documents_chain(llm, prompt)
            rag_chain = create_retrieval_chain(retriever, summary_chain)
           
            response = rag_chain.invoke({"input": chat_query})
            st.success("Summary Generated:")
            st.json(response["answer"])
    except Exception as e:
        st.exception(f'Exception: {e}')
