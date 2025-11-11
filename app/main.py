# app/main.py
from fastapi import FastAPI
from app.api import projects
from app.api import bundles

app = FastAPI(title="Reviewer POC")
app.include_router(projects.router)
app.include_router(bundles.router)

@app.get("/health")
def health():
    return {"status": "ok"}
