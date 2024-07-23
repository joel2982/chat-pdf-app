from fastapi import APIRouter, UploadFile, File, HTTPException
import logging
from services.pdf_processor import process_pdf
from services.llm import ChatBot
from pydantic import BaseModel

class Query(BaseModel):
    text : str

router = APIRouter()
chat_bot = None

@router.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file format")
    global chat_bot
    pdf_content = await file.read()
    processed_text = process_pdf(pdf_content)
    chat_bot = ChatBot(processed_text)
    return {"message": "PDF text added to context"}

@router.post("/chat/")
async def chat(query: Query):
    if chat_bot is None:
        raise HTTPException(status_code=400, detail="Please upload a PDF first")
    try:
        response = chat_bot.get_response(query.text)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail="An error occurred while processing your query")
