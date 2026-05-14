# Conversational AI Chatbot API

This project provides a FastAPI-based chatbot with RAG and Tool-calling capabilities.

## Setup

1. Create a virtual environment and activate it.
2. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the root directory (where you run the server from) and add your Google API key:
   ```env
   GOOGLE_API_KEY="your-google-api-key-here"
   ```

## Running the Server

Run the FastAPI server using Uvicorn:
```bash
uvicorn chatbot_app.main:app --reload
```
The server will start at `http://127.0.0.1:8000`.

## API Endpoints

### 1. `/upload` (POST)
Upload a document (PDF or TXT) to the vector database.
- **Form Data**: `file` (the file to upload)

Example using `curl`:
```bash
curl -X POST -F "file=@sample.pdf" http://127.0.0.1:8000/upload
```

### 2. `/chat` (POST)
Send a message and get a streaming response. The AI will remember the history based on `session_id`.
- **JSON Body**: `{"session_id": "user123", "message": "What is the capital of France?"}`

Example using `curl`:
```bash
curl -N -X POST http://127.0.0.1:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"session_id": "user123", "message": "What is the capital of France?"}'
```
*(The `-N` flag is important in curl to see the SSE stream as it happens)*
