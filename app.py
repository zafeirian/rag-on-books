from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from dotenv import load_dotenv
import os
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


from contextlib import asynccontextmanager # for lifespan events
#We need to load the VDB when the application begins and use it for every request (lifespan).

EMBED_MODEL = 'gpt-embedding-3-small'
CHROMA_PATH = 'chroma'

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

    app.state.embedding_function = embedding_function
    app.state.db = db

    try:
        yield # Handling requests here.
    finally:
        app.state.db = None
        app.state.embedding_function = None


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

@app.get("health")
def health():
    return {"ok": True}

@app.post("/query", response_model=QueryRespone)
def query(req: QueryRequest):
    db: 

    results = 