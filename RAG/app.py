from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Welcome to the RAG Tool"}

@app.get("/health")
async def health_check():
    return {"status": "ok"}