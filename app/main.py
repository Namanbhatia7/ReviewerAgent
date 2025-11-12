# app/main.py
from fastapi import FastAPI
from app.api import projects
from app.api import bundles

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Reviewer POC")
app.include_router(projects.router)
app.include_router(bundles.router)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}
