import re
import pdfplumber
import io

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Extracts raw text from a PDF file buffer using pdfplumber.
    """
    extracted_text = ""
    
    # Use io.BytesIO to read the file directly from memory (no saving to disk needed)
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                extracted_text += page_text + "\n\n"
                
    return extracted_text.strip()

def chunk_contract_text_semantically(text: str) -> list[str]:
    """
    Splits the massive legal contract semantically by looking for 
    legal clause markers like 'Article 1', 'Section 2', or '1.1'.
    This ensures legal clauses remain completely intact for ChromaDB.
    """
    # This regex looks for newlines followed by "Article X", "Section X", or numbering like "1.1"
    split_pattern = r'\n(?=(?:Article\s+[IVX\d]+|Section\s+\d+|\d+\.\d*)\b)'
    
    raw_chunks = re.split(split_pattern, text, flags=re.IGNORECASE)
    
    # Clean up the chunks (remove tiny artifacts or empty strings)
    semantic_chunks = []
    for chunk in raw_chunks:
        clean_chunk = chunk.strip()
        # Ignore random tiny fragments (like page numbers or single words)
        if len(clean_chunk) > 30:  
            semantic_chunks.append(clean_chunk)
            
    return semantic_chunks

def process_upload(pdf_bytes: bytes) -> dict:
    """
    Master function to orchestrate the parsing and semantic chunking.
    """
    # 1. Extract the raw text
    raw_text = extract_text_from_pdf(pdf_bytes)
    
    if not raw_text:
        raise ValueError("Could not extract any text from the PDF. The file may be an image/scanned PDF.")
    
    # 2. Chunk the text semantically for the VectorDB
    chunks = chunk_contract_text_semantically(raw_text)
    
    return {
        "raw_text": raw_text,
        "chunks": chunks,
        "total_chunks": len(chunks)
    }