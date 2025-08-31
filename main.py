import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
from langchain_core.documents import Document

DATA_PATH = "data/books/"

def load_documents():
    loader = PyPDFLoader(DATA_PATH + "the-hobbit.pdf")
    documents = loader.load()
    print(len(documents))
    for doc in documents:
        doc.page_content = re.sub(r"\s+", " ", doc.page_content).strip()
    return documents

def split_into_chunks(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=500,
        length_function=len,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

documents = load_documents()
chunks = split_into_chunks(documents)

print(chunks[60].page_content)
print(chunks[60].metadata)