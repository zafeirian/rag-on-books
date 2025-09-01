from fastapi import FastAPI, HTTPException

app = FastAPI(title="Simple RAG on LOTR books.")

@app.get("health")
def health():
    return {"ok": True}

