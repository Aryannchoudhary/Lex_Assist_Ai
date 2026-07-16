import streamlit as st
import requests
import os

BACKEND_URL = os.getenv("BACKEND_URL", "http://lex-assist-backend:8000/api")

# Configure the page layout
st.set_page_config(page_title="Lex-Assist AI", page_icon="⚖️", layout="centered")

# Initialize session state variables
if "contract_id" not in st.session_state:
    st.session_state.contract_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None

st.title("⚖️ Lex-Assist: AI Legal Copilot")

# --- TOP SECTION: Document Upload & Analysis (Inside an Expander) ---
with st.expander("📄 Document Management & Upload", expanded=True):
    st.write("Upload a contract to analyze its risk profile and enable the AI chat.")
    uploaded_file = st.file_uploader("Upload a Legal Contract (PDF)", type=["pdf"], label_visibility="collapsed")
    
    if uploaded_file is not None and st.button("Analyze Contract", use_container_width=True):
        with st.spinner("Extracting text, running ML models, and vectorizing..."):
            # Send the file to the FastAPI backend
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            try:
                response = requests.post(f"{BACKEND_URL}/upload", files=files)
                response.raise_for_status() 
                data = response.json()
                
                # Save the tracking ID and results to session state
                st.session_state.contract_id = data.get("contract_id")
                st.session_state.analysis_results = data
                st.session_state.chat_history = [] # Clear old chat
                
                st.success("Analysis Complete! You can now chat with the document below.")
            
            # --- FIXED: Restored the missing except block ---
            except requests.exceptions.RequestException as e:
                st.error(f"Backend connection failed: {e}")

    # --- FIXED: Removed the duplicate sections and simplified the metric ---
    # Display the ML Risk Analysis directly inside the expander if a document is loaded
    if st.session_state.analysis_results:
        st.divider()
        st.subheader("📊 Risk Assessment")
        res = st.session_state.analysis_results
        
        # Display high-level metrics in a clean row
        col1, col2, col3 = st.columns(3)
        col1.metric("Compliance Score", f"{res.get('overall_compliance_score', 0)}/100")
        col2.metric("High-Risk Clauses", res.get('risk_clauses_found', 0))
        col3.metric("Contract ID", st.session_state.contract_id)

st.divider()

# --- BOTTOM SECTION: The RAG Chatbot ---
if not st.session_state.contract_id:
    st.info("👆 Please upload and analyze a contract in the panel above to begin.")
else:
    # Display the chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input
    if prompt := st.chat_input("Ask a question about this contract..."):
        # 1. Display user message instantly
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # 2. Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # 3. Call the FastAPI backend for the LLM answer
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Consulting contract clauses..."):
                try:
                    payload = {
                        "user_message": prompt,
                        "contract_id": st.session_state.contract_id,
                        "history": [
                            {"role": m["role"], "content": m["content"]} 
                            for m in st.session_state.chat_history[:-1]
                        ] 
                    }
                    
                    response = requests.post(f"{BACKEND_URL}/chat", json=payload)
                    response.raise_for_status()
                    
                    bot_reply = response.json().get("reply", "No response generated.")
                    message_placeholder.markdown(bot_reply)
                    
                    # Add bot response to history
                    st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})
                    
                except requests.exceptions.RequestException as e:
                    st.error(f"Error communicating with AI: {e}")