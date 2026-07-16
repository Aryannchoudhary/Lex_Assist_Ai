import re
import pdfplumber
import io

# Matches a newline followed by a clause/section marker at the start of the next line:
# "Article I", "Section 2", "3.1", "4." etc.
_SECTION_MARKER = re.compile(
    r'\n(?=(?:Article\s+[IVXLCDM\d]+\b|Section\s+\d+(?:\.\d+)*\b|\d+\.\d+\b|\d+\.\s))',
    re.IGNORECASE,
)

MIN_CHUNK_CHARS = 30     # fragments shorter than this get merged into a neighbor instead of dropped
MAX_CHUNK_CHARS = 2000   # roughly the effective context window of the BiLSTM/embedding models downstream


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Extracts raw text from a PDF file buffer using pdfplumber.
    Skips individual pages that fail to parse instead of failing the whole document.
    """
    extracted_text = ""

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            try:
                page_text = page.extract_text()
            except Exception:
                # One malformed page shouldn't kill extraction for the rest of the document
                page_text = None
            if page_text:
                extracted_text += page_text + "\n\n"

    # Collapse excessive whitespace/blank lines left behind by PDF extraction
    extracted_text = re.sub(r'[ \t]+', ' ', extracted_text)
    extracted_text = re.sub(r'\n{3,}', '\n\n', extracted_text)

    return extracted_text.strip()


def _split_oversized_chunk(chunk: str) -> list[str]:
    """
    Breaks a chunk that's too large down further, first by paragraph breaks,
    then by sentence boundaries if a single paragraph is still too big.
    """
    if len(chunk) <= MAX_CHUNK_CHARS:
        return [chunk]

    pieces = []
    for para in re.split(r'\n\s*\n', chunk):
        para = para.strip()
        if not para:
            continue
        if len(para) <= MAX_CHUNK_CHARS:
            pieces.append(para)
        else:
            # Last resort: split on sentence boundaries and regroup up to the size cap
            sentences = re.split(r'(?<=[.!?])\s+', para)
            buffer = ""
            for sentence in sentences:
                if len(buffer) + len(sentence) + 1 > MAX_CHUNK_CHARS and buffer:
                    pieces.append(buffer.strip())
                    buffer = sentence
                else:
                    buffer = f"{buffer} {sentence}".strip()
            if buffer:
                pieces.append(buffer.strip())
    return pieces


def chunk_contract_text_semantically(text: str) -> list[str]:
    """
    Splits the contract text by legal clause markers ('Article 1', 'Section 2', '1.1', etc).
    Falls back to paragraph/sentence-based splitting when no markers are found or a
    matched chunk is too large, and merges undersized fragments into a neighboring
    chunk instead of discarding them.
    """
    if not text:
        return []

    raw_chunks = _SECTION_MARKER.split(text)

    # No markers found at all -> the whole text came back as one chunk.
    # Force it through the size-based splitter so we don't lose most of the document.
    if len(raw_chunks) == 1:
        raw_chunks = _split_oversized_chunk(raw_chunks[0])
    else:
        expanded = []
        for chunk in raw_chunks:
            expanded.extend(_split_oversized_chunk(chunk))
        raw_chunks = expanded

    # Merge undersized fragments into the previous chunk instead of dropping them,
    # so short-but-meaningful text (e.g. signature blocks) isn't lost.
    semantic_chunks: list[str] = []
    for raw in raw_chunks:
        clean_chunk = raw.strip()
        if not clean_chunk:
            continue
        if len(clean_chunk) < MIN_CHUNK_CHARS and semantic_chunks:
            semantic_chunks[-1] = f"{semantic_chunks[-1]}\n{clean_chunk}"
        else:
            semantic_chunks.append(clean_chunk)

    return semantic_chunks


def process_upload(pdf_bytes: bytes) -> dict:
    """
    Master function to orchestrate parsing and semantic chunking.
    Return shape is unchanged: {"raw_text", "chunks", "total_chunks"}.
    """
    raw_text = extract_text_from_pdf(pdf_bytes)

    if not raw_text:
        raise ValueError("Could not extract any text from the PDF. The file may be an image/scanned PDF.")

    chunks = chunk_contract_text_semantically(raw_text)

    return {
        "raw_text": raw_text,
        "chunks": chunks,
        "total_chunks": len(chunks)
    }