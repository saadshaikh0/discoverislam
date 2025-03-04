from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from sqlalchemy.orm import Session
import os
import sys
import asyncio
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from database import get_db
from dao import update_query_response

router = APIRouter(prefix="/modelresponse", tags=["Model Response"])

# ✅ Load Qwen 3B Model and Tokenizer
MODEL_NAME = "Qwen/Qwen-3B"  # Model Name from Hugging Face
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading Qwen 3B Model on {device}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")
print("✅ Qwen Model Loaded Successfully!")

async def generate_response_stream(prompt: str):
    """Generate response using Qwen 3B model and stream chunks."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # ✅ Generate response in streaming mode
    with torch.no_grad():
        outputs = model.generate(input_ids, max_length=512, do_sample=True)

    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # ✅ Send response in chunks
    for chunk in response_text.split(" "):
        yield chunk + " "
        await asyncio.sleep(0.05)  # ✅ Simulate real-time streaming delay

@router.websocket("/stream/{query_id}")
async def stream_response(websocket: WebSocket, query_id: int, db: Session = Depends(get_db)):
    await websocket.accept()

    response_text = ""
    
    # ✅ Fetch query details
    query = db.query(UserQuery).filter(UserQuery.id == query_id).first()
    if not query:
        await websocket.send_text("Query not found.")
        await websocket.close()
        return

    user_email = query.user_email  # ✅ Retrieve user email
    user_session_id = query.user_session_id  # ✅ Retrieve session ID
    print(f"Generating response for query: {query.query}")
    
    try:
        # ✅ Call LLM and Stream Response in Real-Time
        async for chunk in generate_response_stream(query.query):
            await websocket.send_text(chunk)  # ✅ Send chunk immediately
            response_text += chunk

        # ✅ Save full response in DB after streaming completes
        update_query_response(db, query_id, response_text.strip())

        await websocket.send_text("Response Completed ✅")  # Notify frontend
    except WebSocketDisconnect:
        print(f"Client disconnected while streaming response for query {query_id}")
