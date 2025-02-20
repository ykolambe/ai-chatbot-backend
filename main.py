from fastapi import FastAPI
from pydantic import BaseModel
from starlette.responses import StreamingResponse
import boto3
import json
import asyncio
from typing import AsyncGenerator
from bedrock import generate_response

async def stream_response(prompt: str) -> AsyncGenerator[str, None]:
    """Streams the response character by character to mimic ChatGPT's typing effect."""
    response = generate_response(prompt)
    for char in response:
        yield char
        await asyncio.sleep(0.001)  # Small delay for smoother effect

app = FastAPI()

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat(request: ChatRequest):
    return StreamingResponse(stream_response(request.message), media_type="text/plain")


