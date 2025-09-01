from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os
import argparse

load_dotenv()

CHROMA_PATH = 'chroma'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('query_text', type=str, help='The query text.')
    args = parser.parse_args()
    query_text = args.query_text


    embedding_function = OpenAIEmbeddings(model='text-embedding-3-small', api_key=os.getenv('OPENAI_API_KEY'))
    db = Chroma(embedding_function=embedding_function, persist_directory=CHROMA_PATH)

    # Searching the VDB
    results = Chroma._similarity_search_with_relevance_scores(query_text, k=4)
    if len(results) == 0 or results[0][1]<0.7:
        print(f'Unable to find relevant content.')
        return
    
    context_text = '\n\n---\n\n'.join([doc.page_content for doc, _score in results])
    print(context_text)
    