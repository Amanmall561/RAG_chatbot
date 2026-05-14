import os
import json
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables (like GOOGLE_API_KEY)
load_dotenv()

# We import these after load_dotenv to ensure environment variables are present
from .agent import get_agent
from .rag import process_and_store_document

app = FastAPI(title="Conversational AI Chatbot API")

agent_executor = get_agent()

class ChatRequest(BaseModel):
    session_id: str
    message: str

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename missing.")
        
    if not (file.filename.lower().endswith('.pdf') or file.filename.lower().endswith('.txt')):
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported.")
        
    try:
        contents = await file.read()
        num_chunks = process_and_store_document(contents, file.filename)
        return {"message": f"Successfully processed '{file.filename}'. Stored {num_chunks} chunks."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(request: ChatRequest):
    async def event_generator():
        try:
            async for event in agent_executor.astream_events(
                {"messages": [("user", request.message)]},
                config={"configurable": {"thread_id": request.session_id}},
                version="v2"
            ):
                if event["event"] == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    # If it's standard content (not a tool call artifact)
                    if isinstance(chunk.content, str) and chunk.content:
                        # Yield in Server-Sent Events (SSE) format
                        yield f"data: {json.dumps({'content': chunk.content})}\n\n"
            
            # Send a completion event
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("chatbot_app.main:app", host="0.0.0.0", port=8000, reload=True)
