from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from dotenv import load_dotenv
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from contextlib import asynccontextmanager # for lifespan events
#We need to load the VDB when the application begins and use it for every request (lifespan).

EMBED_MODEL = 'gpt-embedding-3-small'
CHROMA_PATH = 'chroma'
LLM_MODEL = 'gpt-4o-mini'
PROMPT_TEMPLATE = '''
Answer the question based ONLY on the following context:

{context}

---

Answer the question based on the above context: {query}
'''

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set.")

    embedding_function = OpenAIEmbeddings(
        model=EMBED_MODEL,
        api_key=api_key,
    )

    db = Chroma(
        embedding_function=embedding_function,
        persist_directory=CHROMA_PATH
    )

    llm = ChatOpenAI(
        model=LLM_MODEL,
        api_key=api_key,
        temperature=0,
    )

    app.state.embedding_function = embedding_function
    app.state.db = db
    app.state.llm = llm

    try:
        yield # Handling requests here.
    finally:
        app.state.db = None
        app.state.embedding_function = None
        app.state.llm = None


class QueryRequest(BaseModel):
    text: str
    k: int = 3

class Chunk(BaseModel):
    content: str
    score: Optional[float] = None
    source: Optional[str] = None
    page: Optional[int] = None

class QueryRespone(BaseModel):
    response: str
    sources: list[str]

app = FastAPI(title="Simple RAG on LOTR books.", lifespan=lifespan)

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/query", response_model=QueryRespone)
def query(req: QueryRequest):
    db: Chroma = app.state.db

    results = db.similarity_search_with_relevance_scores(query=QueryRequest.text, k=QueryRequest.k)
    if len(results)==0 or results[0][1]<0.25:
        return {"Error": "Unable to find relevant content."}
    
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, query=QueryRequest.text)

    llm = ChatOpenAI()

    
