import streamlit as st
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
# For any RAG Applications we need these both modules below
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS

st.header("RAG Groq")

import os
from dotenv import load_dotenv
load_dotenv()
st.secrets["GROQ_API_KEY"]
st.secrets["HUGGING_FACE"]

os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
os.environ['HUGGING_FACE'] = os.getenv("HUGGING_FACE")

groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(groq_api_key = groq_api_key, model_name = "Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Question:{input}

    """
)

def create_vector_embedding():
    if "vectors" not in st.session_state:
        # Load Documents
        st.session_state.loader = PyPDFDirectoryLoader("./research_paper")  # Ensure correct path
        st.session_state.docs = st.session_state.loader.load() 
        
        if not st.session_state.docs:  # Check if documents were loaded
            st.error("No documents found in the directory!")
            return
        
        # Split Documents
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50]) 
        
        if not st.session_state.final_documents:
            st.error("Document splitting failed! No chunks were created.")
            return

        # Initialize Embedding Model
        embedding_model = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")  # Create an instance
        
        # Generate embeddings
        st.session_state.embeddings = [embedding_model.embed_query(doc.page_content) for doc in st.session_state.final_documents]

        if not st.session_state.embeddings or len(st.session_state.embeddings) == 0:
            st.error("Embedding generation failed!")
            return
        
        # Create FAISS Vectorstore
        st.session_state.vectors = FAISS.from_texts(
            [doc.page_content for doc in st.session_state.final_documents],
            embedding_model
        )

        st.success("Vector Database is Ready!")

# Streamlit UI
user_prompt = st.text_input("Enter your Query from the research paper") 

if st.button("Document Embedding"):  # Button to trigger vector embedding
    create_vector_embedding()

import time 

if user_prompt: 
    document_chain = create_stuff_documents_chain(llm,prompt)
    retriever = st.session_state.vectors.as_retriever()
    retriever_chain = create_retrieval_chain(retriever,document_chain)

    start = time.process_time()
    response = retriever_chain.invoke({'input':user_prompt})
    print(f"Response time : {time.process_time()-start}")

    st.write(response['answer'])

    # with a streamlit expander
    with st.expander("Document similarity Search"):
        for i,doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write('----------------')
