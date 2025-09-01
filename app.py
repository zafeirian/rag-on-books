from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


from contextlib import asynccontextmanager # for lifespan events
#We need to load the VDB when the application begins and use it for every request (lifespan).

EMBED_MODEL = 'text=-embedding-3-small'
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
        chunk_size=64,
        show_progress_bar=False
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


app = FastAPI(title="Simple RAG on LOTR books.")

@app.get("health")
def health():
    return {"ok": True}


@app.on_event()
