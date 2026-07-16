from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import pickle
import os
import uuid
import numpy as np
from typing import List, Dict, Optional

# --- 1. LEGACY KERAS IMPORTS ---
from tensorflow.keras.models import model_from_json, Sequential, Model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Internal Service Imports
from app.services.document_parser import process_upload
from app.services.vector_store import VectorStoreService
from app.services.rag_engine import RAGEngine

# Global state for ML models and database connections
ml_models = {}

# POINTING BACK TO YOUR EXISTING .PKL FILE
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "bilstm_clause_pipeline.pkl")


# --- 2. LEGACY NEURAL NETWORK PIPELINE ---
class BiLSTMPipeline:
    """Helper class to glue the saved dictionary pieces back into a working model."""
    def __init__(self, loaded_dict):
        self.tokenizer = loaded_dict['tokenizer']
        self.label_encoder = loaded_dict.get('label_encoder')

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

        # C. Get the max sequence length from the model (default to 250 if dynamic)
        self.maxlen = self.model.input_shape[1] if (self.model.input_shape and len(self.model.input_shape) > 1 and self.model.input_shape[1]) else 250

    def predict(self, texts):
        # Convert text chunks to numbers, pad them, and predict
        seqs = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(seqs, maxlen=self.maxlen, padding='post', truncating='post')
        preds = self.model.predict(padded)
        pred_indices = np.argmax(preds, axis=1)
        
        # Convert numeric indices back to human-readable legal categories
        if self.label_encoder:
            return [str(label) for label in self.label_encoder.inverse_transform(pred_indices)]
        else:
            return [str(idx) for idx in pred_indices]


# --- 3. STARTUP SEQUENCE ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up Lex-Assist Server...")
    
    # A. Load BiLSTM Pipeline from PKL
    try:
        print(f"Loading legacy PKL model from {MODEL_PATH}...")
        with open(MODEL_PATH, "rb") as f:
            loaded_data = pickle.load(f)
            ml_models["clause_classifier"] = BiLSTMPipeline(loaded_data)
        print("✅ BiLSTM pipeline rebuilt and loaded successfully!")
    except Exception as e:
        print(f"⚠️ WARNING: Failed to load BiLSTM model: {e}")
        ml_models["clause_classifier"] = None
        
    # B. Initialize Vector Database
    try:
        ml_models["vector_store"] = VectorStoreService()
        print("✅ ChromaDB engine initialized successfully!")
    except Exception as e:
        print(f"❌ CRITICAL: Failed to initialize ChromaDB: {e}")
        ml_models["vector_store"] = None
        
    # C. Initialize RAG Engine
    try:
        ml_models["rag_engine"] = RAGEngine()
        print("✅ Mistral/Qwen RAG Engine initialized successfully!")
    except Exception as e:
        print(f"❌ CRITICAL: Failed to initialize RAG Engine: {e}")
        ml_models["rag_engine"] = None
    
    yield
    print("Shutting down server, clearing memory...")
    ml_models.clear()

app = FastAPI(title="Lex-Assist AI API", version="1.0.0", lifespan=lifespan)


# --- 4. PYDANTIC SCHEMAS ---
class ChatRequest(BaseModel):
    user_message: str
    contract_id: str
    history: Optional[List[Dict[str, str]]] = []


# --- 5. API ROUTES ---
@app.get("/")
async def health_check():
    return {
        "status": "Active", 
        "classifier": "Loaded" if ml_models.get("clause_classifier") else "Missing",
        "database": "Connected" if ml_models.get("vector_store") else "Disconnected",
        "llm": "Connected" if ml_models.get("rag_engine") else "Disconnected"
    }


@app.post("/api/upload")
async def upload_contract(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    
    pipeline = ml_models.get("clause_classifier")
    v_store = ml_models.get("vector_store")
    
    if not pipeline or not v_store:
        raise HTTPException(status_code=500, detail="Core server engines are not fully loaded.")

    try:
        # 1. Parse PDF text
        pdf_bytes = await file.read()
        parsed_data = process_upload(pdf_bytes)
        chunks = parsed_data["chunks"]
        total_chunks = len(chunks)
        
        # 2. Run Deep Learning Classification
        clean_predictions = pipeline.predict(chunks)
        contract_id = f"DOC_{str(uuid.uuid4())[:8].upper()}"
        
        analyzed_clauses = []
        risk_count = 0
        
        # 3. Process Risks
        for chunk, category in zip(chunks, clean_predictions):
            is_risk = category in ["Liability", "Termination", "Limitation of Liability", "Indemnity"]
            if is_risk:
                risk_count += 1
            analyzed_clauses.append({
                "text": chunk, 
                "classification": category, 
                "requires_review": is_risk
            })
            
        # 4. Save to Vector Database for RAG
        success = v_store.save_contract_chunks(
            chunks=chunks, 
            contract_id=contract_id, 
            classifications=clean_predictions
        )
        if not success:
            raise HTTPException(status_code=500, detail="Database write action failed.")

        # 5. Return structured payload (matching frontend expectations)
        compliance_score = max(0, 100 - int((risk_count / total_chunks) * 100)) if total_chunks > 0 else 100
        
        return {
            "status": "success",
            "contract_id": contract_id,
            "filename": file.filename,
            "overall_compliance_score": compliance_score,
            "risk_clauses_found": risk_count,
            "analysis_results": analyzed_clauses
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Upload processing failed: {str(e)}")


@app.post("/api/chat")
async def chat_with_contract(request: ChatRequest):
    v_store = ml_models.get("vector_store")
    rag = ml_models.get("rag_engine")
    
    if not v_store or not rag:
        raise HTTPException(status_code=500, detail="Database or RAG engine is offline.")
        
    try:
        # 1. Retrieve similar chunks from ChromaDB
        matched_contexts = v_store.query_similar_chunks(
            query_text=request.user_message, 
            contract_id=request.contract_id
        )
        
        # 2. Generate LLM Answer with Chat History
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