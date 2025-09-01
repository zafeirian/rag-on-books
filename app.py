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
    sources: list[dict]

app = FastAPI(title="Simple RAG on LOTR books.", lifespan=lifespan)

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/ask", response_model=QueryRespone)
async def ask(req: QueryRequest):
    db: Chroma = app.state.db
    llm: ChatOpenAI = app.state.llm

    try:
        results = db.similarity_search_with_relevance_scores(query=req.text, k=req.k)
        if len(results)==0 or results[0][1]<0.25:
            return {"Error": "Unable to find relevant content."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    chunks = [Chunk(content=doc.page_content, score=_score, source=doc.metadata['source'], page=doc.metadata['page']) for doc, _score in results]

    context_text = "\n\n---\n\n".join([chunk.content for chunk in chunks])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, query=req.text)

    response_text = await llm.ainvoke(prompt)
    sources = [{"source": chunk.source, "page": chunk.page} for chunk in chunks]

    return QueryRespone(response=response_text, sources=sources)

    

    
