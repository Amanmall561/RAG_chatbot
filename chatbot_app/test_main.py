import io
import json
import pytest
import sys
from unittest.mock import patch, MagicMock, AsyncMock

# Mock dependencies before importing app
sys.modules['langchain_google_genai'] = MagicMock()
sys.modules['langgraph'] = MagicMock()
sys.modules['langgraph.prebuilt'] = MagicMock()
sys.modules['langgraph.checkpoint.memory'] = MagicMock()

with patch('chatbot_app.main.get_agent') as mock_get_agent, \
     patch('chatbot_app.main.process_and_store_document') as mock_process:
    
    # Mocking the agent to return a fake stream
    class FakeAgent:
        async def astream_events(self, input_dict, config, version):
            # simulate a tool call or simple response
            yield {
                "event": "on_chat_model_stream",
                "data": {
                    "chunk": MagicMock(content="Hello! ")
                }
            }
            yield {
                "event": "on_chat_model_stream",
                "data": {
                    "chunk": MagicMock(content="How can I help you?")
                }
            }

    mock_get_agent.return_value = FakeAgent()
    mock_process.return_value = 5 # 5 chunks

    from chatbot_app.main import app

client = TestClient(app)

def test_upload_document_valid_pdf():
    file_content = b"fake pdf content"
    response = client.post(
        "/upload",
        files={"file": ("test.pdf", file_content, "application/pdf")}
    )
    assert response.status_code == 200
    assert "Stored 5 chunks" in response.json()["message"]

def test_upload_document_invalid_extension():
    file_content = b"fake image content"
    response = client.post(
        "/upload",
        files={"file": ("test.png", file_content, "image/png")}
    )
    assert response.status_code == 400
    assert "Only PDF and TXT files are supported" in response.json()["detail"]

def test_chat_endpoint():
    payload = {
        "session_id": "test_session",
        "message": "Hi"
    }
    response = client.post("/chat", json=payload)
    assert response.status_code == 200
    
    # Check SSE format
    content = response.text
    assert "data: " in content
    assert "[DONE]" in content
    
    # Extract the json content
    lines = content.strip().split('\n')
    chunks = []
    for line in lines:
        if line.startswith("data: ") and not line.endswith("[DONE]"):
            data_str = line[6:]
            try:
                data = json.loads(data_str)
                if "content" in data:
                    chunks.append(data["content"])
            except:
                pass
    
    assert "".join(chunks) == "Hello! How can I help you?"
