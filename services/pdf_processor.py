from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import pickle
import os


def pdf_to_txt(pdf):
    pdf_reader = PdfReader(pdf)
    text = ''
    title = pdf.name[:-4]
    for page in pdf_reader.pages:
            text += page.extract_text()
    return title,text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 2000,
        chunk_overlap = 150
    )
    chunks = text_splitter.split_text(text)
    return chunks

class EmbeddingsGenerator:
    def __init__(self,model_name:str = 'paraphrase-MiniLM-L6-v2') -> None:
        self.__model = SentenceTransformer(model_name)

    def generate_embeddings(self,pdf) :
        title,text = pdf_to_txt(pdf)
        pickle_file_path = f'ChatApp\embeddings\{title}.pkl'
        if os.path.exists(pickle_file_path):
            with open(pickle_file_path,'rb') as f:
                vectorstore = pickle.load(f)
        else:
            chunks = get_text_chunks(text)
            vectorstore = self.__model.encode(chunks).tolist()
            with open(pickle_file_path,'wb') as f:
                pickle.dump(vectorstore,f)
        return vectorstore
