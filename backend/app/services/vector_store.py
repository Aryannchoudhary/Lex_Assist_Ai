import os
import uuid
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings

CHROMA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "db", "chroma_store")

class VectorStoreService:
    def __init__(self):
        os.makedirs(CHROMA_DIR, exist_ok=True)
        self.client = chromadb.PersistentClient(path=CHROMA_DIR)
        
        self.collection = self.client.get_or_create_collection(
            name="legal_contracts",
            metadata={"hnsw:space": "cosine"} 
        )
        
        # CHANGED: Run Hugging Face embeddings locally
        print("Loading local Hugging Face embedding model...")
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    def save_contract_chunks(self, chunks: list[str], contract_id: str, classifications: list[str]) -> bool:
        if not chunks: return False
        
        # The local HF model will embed these seamlessly
        vectors = self.embeddings.embed_documents(chunks)
        
        ids, metadatas = [], []
        for i, (chunk, classification) in enumerate(zip(chunks, classifications)):
            ids.append(f"{contract_id}_chunk_{i}_{str(uuid.uuid4())[:8]}")
            metadatas.append({
                "contract_id": contract_id,
                "clause_index": i,
                "bilstm_classification": classification
            })
            
        self.collection.add(embeddings=vectors, documents=chunks, metadatas=metadatas, ids=ids)
        return True

    def query_similar_chunks(self, query_text: str, contract_id: str, limit: int = 4) -> list[dict]:
        query_vector = self.embeddings.embed_query(query_text)
        
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=limit,
            where={"contract_id": contract_id}
        )
        
        formatted_results = []
        if results and results["documents"]:
            for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                formatted_results.append({
                    "text": doc,
                    "clause_type": meta.get("bilstm_classification", "Unknown")
                })
        return formatted_results