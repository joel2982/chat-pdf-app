from sentence_transformers import util

class RAG:
    def __init__(self, embedding_generator, llm):
        self.embedding_generator = embedding_generator
        self.llm = llm
        self.embeddings = []
        self.texts = []

    def add_documents(self, pdf):
        title,text = self.embedding_generator.pdf_to_txt(pdf)
        chunks = self.embedding_generator.get_text_chunks(text)
        embeddings = self.embedding_generator.generate_embeddings(chunks)
        self.embeddings.extend(embeddings)
        self.texts.extend(text)

    def generate_response(self, user_query: str) -> str:
        query_embedding = self.embedding_generator.generate_embeddings(user_query)[0]
        scores = util.dot_score(query_embedding, self.embeddings)[0].tolist()
        relevant_text = self.texts[scores.index(max(scores))]
        prompt = f"{relevant_text}\n\nUser: {user_query}\nAI:"
        return self.llm.query(prompt)
