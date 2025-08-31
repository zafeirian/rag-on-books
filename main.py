import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader

DATA_PATH = "data/books/"

loader = PyPDFLoader(DATA_PATH + "the_hobbit.pdf")

docs = loader.load()
pages = []
pages.append({"pages": len(docs)})

for doc in docs:
    pages.append(doc.page_content)

print(f"Pages: {pages[0]}")
print(pages[35])
print(pages[36])

def load_documents():
    loader = PyPDFLoader(DATA_PATH + "*.pdf")
    documents = loader.load()
    return documents

