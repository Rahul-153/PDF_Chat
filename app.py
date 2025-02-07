from flask import Flask, request, jsonify
from langchain.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd
from pypdf import PdfReader
import pdfplumber
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os
from flask_cors import CORS


import logging
import sys

# Configure logging
logging.basicConfig(level=logging.DEBUG)
app = Flask(__name__)
app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.DEBUG)
CORS(app,resources={r"/*": {"origins": "*"}})
app.logger.debug("<----------------------Starting the app---------------------->")
app.logger.debug("<----------------------Initialisation---------------------->")

# Initialize embeddings and vector store
os.environ["GROQ_API_KEY"]="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"
llm = ChatGroq(temperature=0, model_name="llama3-8b-8192")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_store = Chroma(embedding_function=embeddings)

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
You should simply write the answer no additional text like based on the information etc.
"""

prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])

retriever = vector_store.as_retriever()
qa = RetrievalQA.from_chain_type(llm=llm,
                            chain_type="stuff",
                            retriever=retriever,
                            return_source_documents=True,
                            chain_type_kwargs={"prompt": prompt})

app.logger.debug("<----------------------Ending Initialisation---------------------->")

@app.route('/upload', methods=['POST'])
def upload_pdf():
    app.logger.debug("Entered upload_pdf")
    """
    Endpoint to upload a PDF, extract text, and store embeddings.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty file name"}), 400

    # Save the file temporarily
    file_path = os.path.join("temp", file.filename)
    file.save(file_path)

    # Extract text and store embeddings
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)
    vector_store.add_documents(documents=all_splits)

    # Cleanup
    os.remove(file_path)

    return jsonify({"message": "PDF uploaded and processed successfully."}), 200

@app.route('/ask', methods=['POST'])
def ask_question():
    """
    Endpoint to handle user queries and return responses.
    """
    app.logger.debug(request.json)
    data = request.json
    query = data.get("query", "")
    if not query:
        return jsonify({"error": "Query is required"}), 400

    # Get the response
    response = qa.invoke({'query':query})
    return jsonify({"response": response['result']})

if __name__ == "__main__":
    # Ensure directories exist
    os.makedirs("temp", exist_ok=True)
    os.makedirs("chroma_db", exist_ok=True)

    app.run(debug=True)