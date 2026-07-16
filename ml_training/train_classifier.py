import os
import pickle
# ... existing imports ...

def train():
    # ... [Your existing data loading, tokenization, and model.fit() logic] ...

    # 1. Save the neural network using the native, version-safe Keras format
    print("💾 Saving the BiLSTM model natively...")
    keras_model_path = "ml_training/saved_models/bilstm_model.keras"
    os.makedirs(os.path.dirname(keras_model_path), exist_ok=True)
    model.save(keras_model_path)

    # 2. Pickle ONLY the NLP metadata (Tokenizer, sequence length, and LabelEncoder if used)
    print("💾 Pickling the NLP metadata...")
    metadata = {
        "tokenizer": tokenizer,
        "max_sequence_length": MAX_SEQUENCE_LENGTH
        # "label_encoder": label_encoder  <-- Include this if you are using one
    }
    
    metadata_path = "ml_training/saved_models/nlp_metadata.pkl"
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
        
    print("✅ Success! Model and metadata saved safely.")