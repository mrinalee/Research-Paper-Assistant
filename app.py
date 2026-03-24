from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from typing import List
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

# Global storage
vector_db = None
retriever = None

# LLM
llm = ChatGroq(
    temperature=0,
    model="openai/gpt-oss-120b"
)

# Prompt
prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant.

Answer in 1-2 sentences only.
Use only the context.

Context:
{context}

Question:
{input}

Answer:
""")

# Health check
@app.get("/health")
def health():
    return {"status": "running"}

# Upload PDF
#@app.post("/upload")
# async def upload_pdf(file: UploadFile = File(...)):
    # global vector_db, retriever

    # file_path = f"data/{file.filename}"

    # with open(file_path, "wb") as f:
    #     f.write(await file.read())

    # # Load PDF
    # loader = PyMuPDFLoader(file_path)
    # docs = loader.load()

    # # Add metadata
    # for doc in docs:
    #     doc.metadata["paper"] = file.filename
    #     doc.metadata["page"] = doc.metadata.get("page", "N/A")

    # # Chunking
    # splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=500,
    #     chunk_overlap=100
    # )
    # chunks = splitter.split_documents(docs)

    # # Embeddings
    # embedding = HuggingFaceEmbeddings(
    #     model_name="sentence-transformers/all-MiniLM-L6-v2"
    # )

    # # Create / Update vector DB
    # if vector_db is None:
    #     vector_db = FAISS.from_documents(chunks, embedding)
    # else:
    #     vector_db.add_documents(chunks)

    # # Retriever
    # retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    # return {"message": f"{file.filename} uploaded successfully"}


@app.post("/upload")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    global vector_db, retriever

    all_chunks = []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    for file in files:
        file_path = f"data/{file.filename}"

        # Save file
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Load PDF
        loader = PyMuPDFLoader(file_path)
        docs = loader.load()

        # Add metadata
        for doc in docs:
            doc.metadata["paper"] = file.filename
            doc.metadata["page"] = doc.metadata.get("page", "N/A")

        # Chunking
        chunks = splitter.split_documents(docs)
        all_chunks.extend(chunks)

    # Create / Update vector DB
    if vector_db is None:
        vector_db = FAISS.from_documents(all_chunks, embedding)
    else:
        vector_db.add_documents(all_chunks)

    # Retriever
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    return {"message": f"{len(files)} files uploaded successfully"}

# Chat endpoint
@app.post("/chat")
async def chat(query: str):
    global retriever

    if retriever is None:
        return JSONResponse(
            status_code=400,
            content={"error": "Upload PDF first"}
        )

    document_chain = create_stuff_documents_chain(llm, prompt)
    qa_chain = create_retrieval_chain(retriever, document_chain)

    response = qa_chain.invoke({"input": query})

    answer = response["answer"]
    sources = response.get("context", [])

    # Get one source
    source_info = "Not available"
    if sources:
        doc = sources[0]
        source_info = f"{doc.metadata.get('paper')} - Page {doc.metadata.get('page')}"

    return {
        "question": query,
        "answer": answer,
        "source": source_info
    }