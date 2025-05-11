# Edvance-AI-tutor

Academic terminal chatbot for answering questions using university course notes in PDF format.

## Features
- Loads and processes university notes from PDF files.
- Splits notes into overlapping chunks based on token count (not word count) for optimal embedding and retrieval.
- Uses Google Gemini API for both embeddings and generative answers.
- Supports semantic search over notes using vector similarity.
- Answers user questions using the most relevant note chunks, or general knowledge if not found in notes.

## Setup

1. **Clone the repository** and navigate to the project directory.

2. **Install dependencies** (Python 3.9+ recommended):
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your Gemini API key:**
   - Create a `.env.local` file in the `Edvance-AI-tutor` directory with the following content:
     ```
     GEMINI-API-KEY=your_google_gemini_api_key_here
     ```

4. **Prepare your notes:**
   - Place your course PDF notes in the appropriate directory, e.g.:
     `../Drive/University-Notes-Repository/ACP/ACP-notes/`

## Usage

Run the chatbot from the terminal:
```bash
python chatbot.py
```

- Enter the course code (e.g., `ACP`) when prompted.
- The chatbot will extract text from PDFs, chunk the text by tokens, embed the chunks, and allow you to ask questions.
- Type `exit` to quit.

## Technical Details

- **Chunking:**
  - Notes are split into overlapping chunks using token counts (default: 1000 tokens per chunk, 100 token overlap).
  - Tokenization uses the `tiktoken` library if available, otherwise falls back to word count.
  - Chunks smaller than 50 tokens are skipped.

- **Embeddings:**
  - Uses Gemini's `models/embedding-001` for chunk/document embeddings (`task_type='retrieval_document'`).
  - Uses Gemini's `models/embedding-001` for query embeddings (`task_type='retrieval_query'`).
  - Embeddings are compared using cosine similarity to find the most relevant chunks.

- **Answer Generation:**
  - The top-k most relevant chunks are provided as context to Gemini's generative model (`models/gemini-1.5-pro-latest`).
  - If the answer is not found in the notes, Gemini is instructed to use general knowledge.

## Troubleshooting
- Ensure your `.env.local` file is present and contains a valid Gemini API key.
- Make sure your notes are in the correct directory and are readable PDFs.
- If you encounter API errors, check your API key and rate limits.

## Dependencies
- PyMuPDF
- python-dotenv
- google-generativeai
- scikit-learn
- numpy
- tiktoken (optional, for accurate token counting)

## License
MIT