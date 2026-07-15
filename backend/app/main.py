from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import pickle
import os
import uuid
import numpy as np
from typing import List, Dict, Optional  # Added typing for the history list

# Bring in Keras to rebuild the dictionary pieces
from tensorflow.keras.models import model_from_json, Sequential, Model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- 1. THE IMPORTS ---
from app.services.document_parser import process_upload
from app.services.vector_store import VectorStoreService
from app.services.rag_engine import RAGEngine

ml_models = {}
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "bilstm_clause_pipeline.pkl")


# --- 2. REBUILDING THE NEURAL NETWORK ---
class BiLSTMPipeline:
    """Helper class to glue the saved dictionary pieces back into a working model."""
    def __init__(self, loaded_dict):
        self.tokenizer = loaded_dict['tokenizer']
        self.label_encoder = loaded_dict['label_encoder']

        # A. Rebuild the Keras Architecture
        config = loaded_dict['model_config']
        if isinstance(config, str):
            self.model = model_from_json(config)
        else:
            try:
                # 99% of beginner models are Sequential
                self.model = Sequential.from_config(config)
            except Exception:
                # Modern Keras 3 fallback for functional models
                self.model = Model.from_config(config)

        # B. Load the trained weights
        self.model.set_weights(loaded_dict['model_weights'])

        # C. Get the max sequence length from the model (default to 200 if dynamic)
        self.maxlen = self.model.input_shape[1] if self.model.input_shape[1] else 200

    def predict(self, texts):
        # 1. Convert text chunks to numbers using the Tokenizer
        seqs = self.tokenizer.texts_to_sequences(texts)
        # 2. Pad the sequences so they are all the exact same length
        padded = pad_sequences(seqs, maxlen=self.maxlen, padding='post', truncating='post')
        # 3. Get probabilities from the neural network
        preds = self.model.predict(padded)
        # 4. Find the highest probability for each chunk
        pred_indices = np.argmax(preds, axis=1)
        # 5. Convert indices back to human-readable strings (e.g., "Liability")
        return [str(label) for label in self.label_encoder.inverse_transform(pred_indices)]


# --- 3. THE STARTUP SEQUENCE ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up server...")
    
    try:
        with open(MODEL_PATH, "rb") as f:
            loaded_data = pickle.load(f)
            ml_models["clause_classifier"] = BiLSTMPipeline(loaded_data)
        print("BiLSTM pipeline rebuilt and loaded successfully!")
    except Exception as e:
        print(f"WARNING: Failed to load BiLSTM model: {e}")
        ml_models["clause_classifier"] = None
        
    try:
        ml_models["vector_store"] = VectorStoreService()
        print("ChromaDB engine initialized successfully!")
    except Exception as e:
        print(f"CRITICAL: Failed to initialize ChromaDB: {e}")
        ml_models["vector_store"] = None
        
    try:
        ml_models["rag_engine"] = RAGEngine()
        print("Mistral/Qwen RAG Engine initialized successfully!")
    except Exception as e:
        print(f"CRITICAL: Failed to initialize RAG Engine: {e}")
        ml_models["rag_engine"] = None
    
    yield
    print("Shutting down server...")
    ml_models.clear()

app = FastAPI(title="Lex-Assist AI API", version="1.0.0", lifespan=lifespan)


# THE UPGRADE: Added history to the request model
class ChatRequest(BaseModel):
    user_message: str
    contract_id: str
    history: Optional[List[Dict[str, str]]] = []


@app.get("/")
async def health_check():
    return {
        "status": "Active", 
        "classifier": "Loaded" if ml_models.get("clause_classifier") else "Missing",
        "database": "Connected" if ml_models.get("vector_store") else "Disconnected",
        "llm": "Connected" if ml_models.get("rag_engine") else "Disconnected"
    }

# --- 4. THE ROUTES ---
@app.post("/api/upload")
async def upload_contract(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    
    pipeline = ml_models.get("clause_classifier")
    v_store = ml_models.get("vector_store")
    
    if not pipeline or not v_store:
        raise HTTPException(status_code=500, detail="Core server engines are not fully loaded.")

    try:
        pdf_bytes = await file.read()
        parsed_data = process_upload(pdf_bytes)
        chunks = parsed_data["chunks"]
        
        clean_predictions = pipeline.predict(chunks)
        contract_id = f"DOC_{str(uuid.uuid4())[:8].upper()}"
        
        analyzed_clauses = []
        risk_count = 0
        for chunk, category in zip(chunks, clean_predictions):
            is_risk = category in ["Liability", "Termination", "Limitation of Liability", "Indemnity"]
            if is_risk:
                risk_count += 1
            analyzed_clauses.append({"text": chunk, "classification": category, "requires_review": is_risk})
            
        success = v_store.save_contract_chunks(
            chunks=chunks, 
            contract_id=contract_id, 
            classifications=clean_predictions
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Database write action failed.")

        total_chunks = len(chunks)
        return {
            "contract_id": contract_id,
            "filename": file.filename,
            "overall_compliance_score": max(0, 100 - int((risk_count / total_chunks) * 100)) if total_chunks > 0 else 100,
            "analysis_results": analyzed_clauses
        }

    except Exception as e:
        import traceback
        traceback.print_exc()  # Force Uvicorn to print the stack trace!
        raise HTTPException(status_code=500, detail=f"Upload processing failed: {str(e)}")


@app.post("/api/chat")
async def chat_with_contract(request: ChatRequest):
    v_store = ml_models.get("vector_store")
    rag = ml_models.get("rag_engine")
    
    if not v_store or not rag:
        raise HTTPException(status_code=500, detail="Database or RAG engine is offline.")
        
    try:
        matched_contexts = v_store.query_similar_chunks(
            query_text=request.user_message, 
            contract_id=request.contract_id
        )
        
        # THE UPGRADE: Passing the history to the LLM
        llm_response = rag.generate_answer(
            query=request.user_message,
            retrieved_blocks=matched_contexts,
            history=request.history 
        )
        
        return {
            "query": request.user_message,
            "contract_id": request.contract_id,
            "reply": llm_response
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat generation failed: {str(e)}")