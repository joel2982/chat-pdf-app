from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
from services.pdf_processor import EmbeddingGenerator
from services.llm import LLMClient
from services.rag import RAG

router = APIRouter()

# Initialize components
embedding_generator = EmbeddingGenerator()
llm_client = LLMClient(model_name="gpt2")
rag = RAG(embedding_generator=embedding_generator, llm=llm_client)

@router.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file format")
    rag.add_documents(await file.read())
    return {"message": "PDF text added to context"}

@router.post("/chat/")
async def chat_with_pdf(query: str):
    response = rag.generate_response(user_query=query)
    return {"response": response}
