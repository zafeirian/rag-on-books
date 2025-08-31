import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
import shutil
import glob

DATA_PATH = "data/books/"
CHROMA_PATH = "chroma"
load_dotenv()
embedding_function = OpenAIEmbeddings(model='gpt-4o-mini', api_key=os.getenv("OPENAI_API_KEY"))

def load_documents():
    all_docs = []
    for pdf_file in glob.glob(DATA_PATH + "*.pdf"):
        loader = PyPDFLoader(pdf_file)
        documents = loader.load()
        for doc in documents:
            doc.page_content = re.sub(r"\s+", " ", doc.page_content).strip()
        all_docs.extend(documents)
    return all_docs

def split_into_chunks(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=500,
        length_function=len,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

def create_vdb(chunks: list[Document], embedding_function = embedding_function):
    # Deleting previous VDBases
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    db = Chroma.from_documents(
            documents=chunks, embedding=embedding_function, persist_directory=CHROMA_PATH
        )
    db.persist()
    print(f"Saves {len(chunks)} chunks into a VDB.")

def generate_vdb():
    documents = load_documents()
    chunks = split_into_chunks(documents)
    create_vdb(chunks)

if __name__ == "__main__":
    generate_vdb()
