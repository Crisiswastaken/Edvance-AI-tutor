import os
import sys
import time
import fitz  # PyMuPDF
import google.generativeai as genai
from dotenv import load_dotenv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Try to import tiktoken for token counting, else fallback to word count
def count_tokens(text):
    try:
        import tiktoken
        enc = tiktoken.get_encoding('cl100k_base')
        return len(enc.encode(text))
    except Exception:
        return len(text.split())

# --- CONFIG ---
CHUNK_SIZE = 1000 #tokens
CHUNK_OVERLAP = 100 #tokens
EMBED_MODEL = 'models/embedding-001'  # Gemini 1.5 Pro embedding model
GEN_MODEL = 'models/gemini-1.5-pro-latest'
API_DELAY = 30  # seconds

# --- ENVIRONMENT ---
load_dotenv('.env.local')
GEMINI_API_KEY = os.getenv('GEMINI-API-KEY')
if not GEMINI_API_KEY:
    print("Error: GEMINI-API-KEY not found in .env.local file")
    sys.exit(1)
genai.configure(api_key=GEMINI_API_KEY)

# --- PDF TEXT EXTRACTION ---
def extract_text_from_pdfs(notes_dir):
    print(f"[DEBUG] Extracting text from PDFs in: {notes_dir}")
    corpus = []
    pdf_files = [f for f in os.listdir(notes_dir) if f.lower().endswith('.pdf')]
    print(f"[DEBUG] Found {len(pdf_files)} PDF files: {pdf_files}")
    if not pdf_files:
        print("[DEBUG] No PDF files found.")
        return None, []
    unreadable = []
    for pdf in pdf_files:
        pdf_path = os.path.join(notes_dir, pdf)
        try:
            print(f"[DEBUG] Opening PDF: {pdf_path}")
            doc = fitz.open(pdf_path)
            text = "\n".join(page.get_text() for page in doc)
            if text.strip():
                corpus.append(text)
                print(f"[DEBUG] Successfully extracted text from: {pdf}")
            else:
                print(f"[DEBUG] No text found in: {pdf}")
                unreadable.append(pdf)
        except Exception as e:
            print(f"[DEBUG] Exception reading {pdf}: {e}")
            unreadable.append(pdf)
    print(f"[DEBUG] Extraction complete. Readable: {len(corpus)}, Unreadable: {len(unreadable)}")
    return "\n".join(corpus), unreadable

# --- CHUNKING ---
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    print(f"[DEBUG] Chunking text with chunk_size={chunk_size}, overlap={overlap}")
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i:i+chunk_size]
        chunk_text_str = " ".join(chunk_words)
        token_count = count_tokens(chunk_text_str)
        if token_count < 50:
            print(f"[DEBUG] Last chunk too small (tokens={token_count}), breaking loop at i={i}")
            break
        chunks.append(chunk_text_str)
        i += chunk_size - overlap
    print(f"[DEBUG] Created {len(chunks)} chunks")
    return [c for c in chunks if c.strip() and count_tokens(c) > 30]

# --- EMBEDDING ---
def embed_texts(texts):
    print(f"[DEBUG] Embedding {len(texts)} text chunks...")
    embeddings = []
    for i, text in enumerate(texts):
        print(f"[DEBUG] Embedding chunk {i+1}/{len(texts)}")
        try:
            result = genai.embed_content(
                model=EMBED_MODEL,
                content=text,
                task_type="retrieval_document",
                title=f"Embedding chunk {i+1}"
            )
            print(f"[DEBUG] Response from embed_content: {result}")
            emb = result['embedding']
            embeddings.append(emb)
            print(f"[DEBUG] Successfully embedded chunk {i+1}")
        except Exception as e:
            print(f"Warning: Failed to embed chunk {i+1}: {e}")
            embeddings.append(None)
        time.sleep(API_DELAY)
    print(f"[DEBUG] Embedding complete. {sum(e is not None for e in embeddings)} successful, {sum(e is None for e in embeddings)} failed.")
    return embeddings

def embed_query(text):
    print(f"[DEBUG] Embedding query: {text}")
    try:
        result = genai.embed_content(
            model=EMBED_MODEL,
            content=text,
            task_type="retrieval_query"
        )
        print(f"[DEBUG] Query embedding response: {result}")
        return result['embedding']
    except Exception as e:
        print(f"Error: Gemini API failed: {e}")
        sys.exit(1)

def get_top_k_chunks(query_emb, chunk_embs, k=5):
    valid = [(i, emb) for i, emb in enumerate(chunk_embs) if emb is not None]
    if not valid:
        return []
    idxs, embs = zip(*valid)
    arr = np.array(embs)
    q = np.array(query_emb).reshape(1, -1)
    sims = cosine_similarity(arr, q).flatten()
    top_k_idx = np.argsort(sims)[-k:][::-1]
    return [idxs[i] for i in top_k_idx]

# --- MAIN CHATBOT LOOP ---
def main():
    while True:
        course_code = input("Enter course code (e.g., CS101): ").strip().upper()
        if not course_code:
            continue
        notes_dir = os.path.join("..", "Drive", "University-Notes-Repository", course_code, f"{course_code}-notes")
        print(f"[DEBUG] Notes directory resolved to: {notes_dir}")
        if not os.path.exists(notes_dir):
            print("Error: Notes folder not found.")
            continue
        print("[DEBUG] Starting PDF extraction...")
        text, unreadable = extract_text_from_pdfs(notes_dir)
        if unreadable:
            for pdf in unreadable:
                print(f"Warning: Could not read {pdf}, skipping.")
        if not text or not text.strip():
            print("Error: No valid notes could be loaded.")
            sys.exit(1)
        print("[DEBUG] Starting chunking...")
        chunks = chunk_text(text)
        if not chunks:
            print("Error: No valid notes could be loaded.")
            sys.exit(1)
        print(f"Loaded {len(chunks)} chunks from notes. Generating embeddings (this may take a while)...")
        print("[DEBUG] Starting embedding of chunks...")
        chunk_embs = embed_texts(chunks)
        if not any(chunk_embs):
            print("Error: No valid notes could be loaded.")
            sys.exit(1)
        print(f"Notes loaded successfully for {course_code} - NOTES.")
        print("You can now ask your questions. Type 'exit' to quit.")
        while True:
            q = input("Your question: ").strip()
            if not q:
                continue
            if q.lower() == 'exit':
                print("Goodbye!")
                sys.exit(0)
            print(f"[DEBUG] Embedding user query...")
            query_emb = embed_query(q)
            print(f"[DEBUG] Getting top k chunks...")
            top_idxs = get_top_k_chunks(query_emb, chunk_embs, k=5)
            if not top_idxs:
                print("Error: No valid notes could be loaded.")
                sys.exit(1)
            prompt = (
                "You are an academic assistant. Use the following notes to answer the user's question.\n\n"
                "Notes:\n" +
                "\n\n".join(chunks[i] for i in top_idxs) +
                f"\n\nQuestion: {q}\n\nIf the answer cannot be found in the notes, infer the answer using your general knowledge or external sources."
            )
            try:
                print(f"[DEBUG] Sending prompt to generative model...")
                model = genai.GenerativeModel(GEN_MODEL)
                response = model.generate_content(prompt)
                print("\n" + response.text.strip() + "\n")
            except Exception as e:
                print(f"Error: Gemini API failed: {e}")
                sys.exit(1)
            print(f"[DEBUG] Waiting for API delay...")
            time.sleep(API_DELAY)

if __name__ == "__main__":
    main() 