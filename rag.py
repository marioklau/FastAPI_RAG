from pinecone_client import retriever
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

EXAM_PROMPT_STRICT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Anda adalah AI pembuat soal ujian.

⚠️ ATURAN KERAS:
- GUNAKAN HANYA informasi dari KONTEKS
- DILARANG menggunakan pengetahuan umum
- Jika konteks TIDAK MEMUAT materi yang diminta → KEMBALIKAN ERROR

### KONTEKS
{context}
### AKHIR KONTEKS

### PERMINTAAN
{question}

Jawab dalam JSON.
"""
)

llm = ChatOpenAI(
    model="gpt-4o-mini-2024-07-18",
    temperature=0.3
)

def validate_context(docs):
    if not docs:
        return False, "Materi tidak ditemukan dalam dokumen."

    text = " ".join([doc.page_content for doc in docs])
    if len(text) < 300:
        return False, "Konteks terlalu sedikit."

    return True, text

def generate_exam(question):
    docs = retriever.invoke(question)

    is_valid, context = validate_context(docs)
    if not is_valid:
        return {"error": context}

    prompt = EXAM_PROMPT_STRICT.format(
        context=context,
        question=question
    )

    response = llm.invoke(prompt)

    return {
        "result": response.content
    }
