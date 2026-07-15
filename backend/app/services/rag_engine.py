import os
from huggingface_hub import InferenceClient

class RAGEngine:
    def __init__(self):
        print("Initializing LLM via Hugging Face native client...")
        
        # Hugging Face's native client automatically handles conversational routing!
        self.client = InferenceClient(
            model="Qwen/Qwen2.5-7B-Instruct",
            # Make sure to put your NEW token here after revoking the old one!
            token="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" 
        )

    def generate_answer(self, query: str, retrieved_blocks: list[dict], history: list[dict] = None) -> str:
        # Initialize memory if this is the first message
        if history is None:
            history = []
            
        if not retrieved_blocks:
            return "No relevant clauses were found in this contract to answer your question."
            
        # Format the context blocks for the LLM
        context_string = ""
        for i, block in enumerate(retrieved_blocks):
            clause_type = block.get('classification', 'Clause')
            context_string += f"\n--- Clause {i+1} (Type: {clause_type}) ---\n{block['text']}\n"
            
        # Build the strict legal prompt manually
        prompt = f"""You are Lex-Assist, an elite AI legal analyst. 
Your job is to answer questions based ONLY on the provided contract clauses.
Do not use outside knowledge. If the answer is not contained in the context, say "The provided contract does not specify this."

CONTRACT CONTEXT:
{context_string}

USER QUESTION:
{query}

Answer precisely and professionally:"""

        # Build the message array with memory!
        messages = [{"role": "system", "content": "You are a precise, professional legal AI."}]
        
        # 1. Add the past conversation
        messages.extend(history)
        
        # 2. Add the current question (with the ChromaDB context injected)
        messages.append({"role": "user", "content": prompt})

        # Run inference using the modern Chat Completion API
        try:
            response = self.client.chat_completion(
                messages=messages,
                max_tokens=512,
                temperature=0.1
            )
            # Extract and return the exact text from the AI's response
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error communicating with AI: {str(e)}"