from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEndpointEmbeddings

class ChatBot:
    def __init__(self, text: str):
        llm_model_name = "mistralai/Mistral-7B-Instruct-v0.2"
        self.llm = HuggingFaceEndpoint(repo_id=llm_model_name,max_length=128,temperature=0.7)        
        # Embedding model
        embedding_model_name = "mixedbread-ai/mxbai-embed-large-v1"
        self.embeddings = HuggingFaceEndpointEmbeddings(model= embedding_model_name,task="feature-extraction")
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_text(text)
        # Create vector store
        self.vectorstore = Chroma.from_texts(texts, self.embeddings)
        # Set up memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        # Set up conversational chain
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            memory=self.memory
        )

    def get_response(self, query: str) -> str:
        try:
            result = self.qa_chain({"question": query})
            print(result)
            return result['answer']
        except Exception:
            return "I'm sorry, I couldn't process your query. Please try again."