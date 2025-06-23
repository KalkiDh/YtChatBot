from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from chatbot_query import create_embedding_chain, create_query_chain, conversation_history, latest_answer

# Initialize FastAPI app
app = FastAPI(title="YouTube Transcript Chatbot API")

# Add CORS middleware to allow requests from all origins/ports
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (e.g., http://localhost:3000, any port)
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # Explicitly allow necessary methods
    allow_headers=["*"],  # Allow all headers
)

# Pydantic models for request validation
class VideoUrlRequest(BaseModel):
    url: str  # Accepts URLs with query params (e.g., &t=311s)

class QueryRequest(BaseModel):
    query: str  # Validates query as a string

# POST endpoint to upload video URL and process transcript
@app.post("/upload-video-url")
async def upload_video(request: VideoUrlRequest):
    embedding_chain = create_embedding_chain()
    try:
        result = embedding_chain.invoke(str(request.url))
        return {
            "status": "success",
            "video_id": result["video_id"],
            "transcript_chunk_count": result["transcript_chunk_count"],
            "first_chunk_excerpt": result["first_chunk_excerpt"],
            "embedding_length": result["embedding_length"],
            "chroma_status": result["chroma_status"]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process video: {str(e)}")

# POST endpoint to send user query
@app.post("/query")
async def send_query(request: QueryRequest):
    query_chain = create_query_chain()
    try:
        result = query_chain.invoke(request.query)
        return {
            "query": result["query"],
            "answer": result["answer"],
            "similar_chunks": result["similar_chunks"],
            "conversation_history": [
                {"role": msg["role"], "content": msg["content"]}
                for msg in result["conversation_history"]
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing query: {str(e)}")

# GET endpoint to retrieve latest answer
@app.get("/response")
async def get_response():
    if not latest_answer:
        raise HTTPException(status_code=404, detail="No response available")
    return {"answer": latest_answer}