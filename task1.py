from pdfminer.high_level import extract_text
import os
import streamlit as st
import pickle
import time
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# Streamlit UI
st.title("Task-1: PDF Scraper with LLM")
st.sidebar.title("PDF Scraper")

uploaded_files = st.sidebar.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
process_pdf_clicked = st.sidebar.button("Process PDFs")
file_path = "faiss_store_openai.pkl"


main_placeholder = st.empty()
llm = ChatGroq(temperature=0, groq_api_key="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", model_name="llama-3.1-70b-versatile")

if process_pdf_clicked:
    st.sidebar.success("Text Extracted Successfully")
    all_text = ""

    # Extract text from all PDFs
    for uploaded_file in uploaded_files:
        extracted_text = extract_text(uploaded_file)
        all_text += extracted_text + "\n"

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    text_chunks = text_splitter.split_text(all_text)

    # Create embeddings and vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore_openai = FAISS.from_texts(text_chunks, embeddings)

    # Save FAISS index
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)

    # Save the FAISS index to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_openai, f)

# Query input
query = main_placeholder.text_input("Ask a Question:")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQA.from_llm(llm=llm, retriever=vectorstore.as_retriever())

        
        # Retrieval-based chain
        retriever = vectorstore.as_retriever()
        chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        # Get response
        result = chain.run(query)

        

        # Display answer
        st.header("Answer")
        st.write(result)

        
