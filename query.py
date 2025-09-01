from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os
import argparse
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

CHROMA_PATH = 'chroma'

PROMPT_TEMP = '''
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {query}
'''

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('query_text', type=str, help='The query text.')
    args = parser.parse_args()
    query_text = args.query_text


    embedding_function = OpenAIEmbeddings(model='text-embedding-3-small', api_key=os.getenv('OPENAI_API_KEY'))
    db = Chroma(embedding_function=embedding_function, persist_directory=CHROMA_PATH)

    # Searching the VDB
    results = db.similarity_search_with_relevance_scores(query=query_text, k=3)

    if len(results) == 0 or results[0][1]<0.25:
        print(f'Unable to find relevant content.')
        return

    context_text = '\n\n---\n\n'.join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMP)
    prompt = prompt_template.format(context=context_text, query=query_text)

    llm = ChatOpenAI(model='gpt-4o-mini', api_key=os.getenv('OPENAI_API_KEY'))
    response_text = llm.invoke(prompt)

    sources = [doc.metadata.get("title", None) for doc, _score in results]
    formatted_response = f"Response: {response_text.content}\nSources: {sources}"
    print(formatted_response)

if __name__ == "__main__":
    main()