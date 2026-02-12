from pinecone_client import retriever
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json
from pinecone_client import retriever
from llm import ask_llm


exam_prompt_text = """
Anda adalah sistem pembuat soal ujian berbasis dokumen (RAG-based exam generator).

TUGAS:
Buat soal sesuai LEVEL TAKSONOMI dan TINGKAT KESULITAN yang diminta pada PERMINTAAN.

====================================================
DEFINISI LEVEL TAKSONOMI (Bloom Revisi)
====================================================

C1 (Mengingat):
- Menguji kemampuan mengingat fakta, istilah, definisi, atau informasi eksplisit.
- Jawaban langsung ditemukan dalam teks.

C2 (Memahami):
- Menguji kemampuan menjelaskan makna, hubungan antar konsep, sebab-akibat,
  atau interpretasi berdasarkan teks.
- Membutuhkan pemahaman, bukan sekadar menyalin kalimat.

====================================================
DEFINISI TINGKAT KESULITAN
====================================================

EASY:
- Berdasarkan satu informasi eksplisit dalam teks.
- Pertanyaan langsung dan tidak membutuhkan penggabungan informasi.

MEDIUM:
- Membutuhkan dua informasi dari teks.
- Menguji hubungan sederhana antar konsep atau detail.

HARD:
- Membutuhkan sintesis ≥ 3 informasi dari konteks.
- Menguji pemahaman implisit atau hubungan kompleks.
- Tidak boleh hanya menyalin satu kalimat.

====================================================
ATURAN KHUSUS UNTUK MATERI NUMERIK
====================================================

Jika dalam konteks terdapat:
- Rumus
- Data angka
- Contoh perhitungan
- Tabel numerik

Maka untuk C2 HARD:
- Soal BOLEH berupa soal perhitungan.
- Jika berupa perhitungan, langkah penyelesaian WAJIB ditampilkan dalam pembahasan.
- Angka yang digunakan HARUS berasal dari konteks.
- Dilarang membuat angka baru di luar konteks.

Jika konteks tidak mengandung data numerik,
maka C2 HARD tetap dibuat dalam bentuk analisis konseptual yang mendalam.

====================================================
ATURAN WAJIB (STRICT RAG MODE)
====================================================

1. Gunakan HANYA informasi dari KONTEKS.
2. Dilarang menggunakan pengetahuan umum.
3. Dilarang mengarang contoh baru yang tidak ada di konteks.
4. Soal HARUS sesuai level_taksonomi dan difficulty yang diminta.
5. Jika informasi tidak cukup → kembalikan FORMAT GAGAL.
6. Output HARUS berupa JSON VALID.
7. Jangan menambahkan teks di luar JSON.

====================================================
VALIDASI INTERNAL SEBELUM MENJAWAB
====================================================

- Periksa apakah informasi dalam konteks cukup untuk membuat soal sesuai level dan difficulty.
- Jika level = C1 HARD → harus menggabungkan beberapa fakta eksplisit.
- Jika level = C2 HARD → harus ada analisis mendalam atau sintesis informasi.
- Jika tidak memenuhi kriteria → kembalikan FORMAT GAGAL.

====================================================
KONTEKS
====================================================
{context}
====================================================
AKHIR KONTEKS
====================================================

PERMINTAAN:
{question}

====================================================
FORMAT JIKA GAGAL
====================================================
{{
  "error": "Materi tidak ditemukan atau tidak cukup untuk membuat soal sesuai level dan difficulty"
}}

====================================================
FORMAT JIKA BERHASIL
====================================================
{{
  "soal": [
    {{
      "level_taksonomi": "C1 atau C2",
      "kategori": "Mengingat atau Memahami",
      "difficulty": "Easy / Medium / Hard",
      "pertanyaan": "...",
      "pilihan": {{
        "A": "...",
        "B": "...",
        "C": "...",
        "D": "..."
      }},
      "jawaban_benar": "A",
      "pembahasan": "Penjelasan harus sepenuhnya berdasarkan konteks."
    }}
  ]
}}
"""

EXAM_PROMPT = PromptTemplate(
    template=exam_prompt_text,
    input_variables=["context", "question"]
)

# ==========================================================
# LLM
# ==========================================================

exam_llm = ChatOpenAI(
    model="gpt-4o-mini-2024-07-18",
    temperature=0
)


# ==========================================================
# VALIDASI CONTEXT
# ==========================================================

def validate_context(docs):
    if not docs:
        return False, "Materi tidak ditemukan dalam dokumen."

    total_text = " ".join([doc.page_content for doc in docs])

    if len(total_text.strip()) < 300:
        return False, "Konteks terlalu sedikit untuk membuat soal."

    return True, total_text


# ==========================================================
# GENERATE EXAM + HALAMAN SUMBER
# ==========================================================

def generate_exam_with_pages(question):

    # 1️⃣ Retrieve
    docs = retriever.invoke(question)

    # 2️⃣ Validasi
    is_valid, context_or_msg = validate_context(docs)
    if not is_valid:
        return {"error": context_or_msg}

    # 3️⃣ Generate pakai prompt STRICT
    chain = (
        EXAM_PROMPT
        | exam_llm
        | StrOutputParser()
    )

    result = chain.invoke({
        "context": context_or_msg,
        "question": question
    })

    # 🔥 Bersihkan jika model kirim ```json
    result = result.strip().replace("```json", "").replace("```", "")

    try:
        parsed_json = json.loads(result)
    except:
        return {"error": "LLM tidak mengembalikan JSON valid"}

    # 🔒 Validasi basic struktur pilihan ganda
    if "soal" not in parsed_json:
        return {"error": "Format tidak sesuai"}

    for item in parsed_json["soal"]:
        if "pilihan" not in item:
            return {"error": "Soal bukan pilihan ganda"}

        if not all(opt in item["pilihan"] for opt in ["A", "B", "C", "D"]):
            return {"error": "Pilihan tidak lengkap"}

    # 4️⃣ Ambil halaman sumber
    pages = set()
    for doc in docs:
        if "page" in doc.metadata:
            pages.add(doc.metadata["page"] + 1)

    return {
        "soal": parsed_json["soal"],
        "halaman_sumber": sorted(list(pages))
    }