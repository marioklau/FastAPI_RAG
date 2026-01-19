from fastapi import FastAPI
from pydantic import BaseModel
from rag import generate_exam

app = FastAPI()

class ExamRequest(BaseModel):
    request: str

@app.post("/generate_exam")
def generate(data: ExamRequest):
    result = generate_exam(data.request)
    return result
