from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from io import BytesIO
import pickle
import os

class EmbeddingGenerator:
    def __init__(self,model_name:str = 'paraphrase-MiniLM-L6-v2') -> None:
        self.embeddings_model = SentenceTransformer(model_name)

    def pdf_to_txt(self,pdf_content):
        pdf_file = BytesIO(pdf_content)
        pdf_reader = PdfReader(pdf_file)
        text = ''
        title = pdf_reader.metadata.title
        for page in pdf_reader.pages:
                text += page.extract_text()
        return title,text

    def get_text_chunks(self,text):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 2000,
            chunk_overlap = 150
        )
        chunks = text_splitter.split_text(text)
        return chunks

    def generate_embeddings(self,texts,title=None):
        pickle_file_path = f'embeddings\{title}.pkl'
        if title and os.path.exists(pickle_file_path):
            with open(pickle_file_path,'rb') as f:
                embeddings = pickle.load(f)
        else:
            embeddings = self.embeddings_model.encode(texts).tolist()
            if title:
                with open(pickle_file_path,'wb') as f:
                    pickle.dump(embeddings,f)
        return embeddings
