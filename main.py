from fastapi import FastAPI
from pydantic import BaseModel
from rag import generate_exam_with_pages   # ⬅️ GANTI INI

app = FastAPI()

class ExamRequest(BaseModel):
    request: str

@app.post("/generate_exam")
def generate(data: ExamRequest):
    result = generate_exam_with_pages(data.request)  # ⬅️ GANTI INI
    return result
