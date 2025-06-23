import os
import re
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain_core.runnables import RunnableLambda, RunnableSequence
from transformers import AutoModel, AutoTokenizer
import torch
import chromadb
from chromadb.utils import embedding_functions

# Load environment variables from .env file
load_dotenv()
hf_api_key = os.getenv("HUGGINGFACE_API_KEY")

if not hf_api_key:
    raise ValueError("HUGGINGFACE_API_KEY not found in .env file")

# Function to extract video ID from YouTube URL
def extract_video_id(url: str) -> dict:
    video_id_pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(video_id_pattern, url)
    if match:
        return {"video_id": match.group(1)}
    else:
        raise ValueError("Invalid YouTube URL")

# Function to fetch transcript
def fetch_transcript(video_id_dict: dict) -> dict:
    try:
        video_id = video_id_dict["video_id"]
        transcript = YouTubeTranscriptApi().fetch(video_id, languages=['en'])
        transcript_text = " ".join([snippet['text'] for snippet in transcript.to_raw_data()])
        return {"video_id": video_id, "transcript": transcript_text}
    except TranscriptsDisabled:
        raise Exception("Transcripts are disabled for this video.")
    except NoTranscriptFound:
        raise Exception("No English transcript found for this video.")
    except Exception as e:
        raise Exception(f"Error fetching transcript: {str(e)}")

# Function to convert text to embeddings
def text_to_embeddings(transcript_dict: dict) -> dict:
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_api_key)
    model = AutoModel.from_pretrained(model_name, token=hf_api_key)
    
    text = transcript_dict["transcript"]
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy().tolist()
    return {
        "video_id": transcript_dict["video_id"],
        "transcript": transcript_dict["transcript"],
        "embeddings": embeddings
    }

# Function to store embeddings in ChromaDB
def store_in_chromadb(data: dict) -> dict:
    client = chromadb.PersistentClient(path="./chroma_db_yt_transcripts")
    collection_name = "youtube_transcripts"
    
    # Create or get collection
    try:
        collection = client.get_collection(collection_name)
    except:
        collection = client.create_collection(
            name=collection_name,
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        )
    
    # Generate a unique ID for the document
    doc_id = f"transcript_{data['video_id']}"
    
    # Store the embedding
    collection.add(
        ids=[doc_id],
        embeddings=[data['embeddings']],
        metadatas=[{
            "video_id": data["video_id"],
            "transcript": data["transcript"][:1000]  # Limit metadata size
        }],
        documents=[data["transcript"]]
    )
    
    return {
        "video_id": data["video_id"],
        "transcript": data["transcript"],
        "embeddings": data["embeddings"],
        "chroma_status": f"Stored transcript for video {data['video_id']} in ChromaDB"
    }

# Function to format output
def format_output(data: dict) -> dict:
    return {
        "video_id": data["video_id"],
        "transcript_excerpt": data["transcript"][:100] + "..." if len(data["transcript"]) > 100 else data["transcript"],
        "embeddings_first_10": data["embeddings"][:10],
        "embedding_length": len(data["embeddings"]),
        "chroma_status": data["chroma_status"]
    }

# Define LangChain chain
def create_transcript_embedding_chain():
    # Step 1: Extract video ID
    extract_video_id_step = RunnableLambda(extract_video_id)
    
    # Step 2: Fetch transcript
    fetch_transcript_step = RunnableLambda(fetch_transcript)
    
    # Step 3: Convert to embeddings
    embeddings_step = RunnableLambda(text_to_embeddings)
    
    # Step 4: Store in ChromaDB
    store_step = RunnableLambda(store_in_chromadb)
    
    # Step 5: Format output
    format_step = RunnableLambda(format_output)
    
    # Create the chain
    chain = RunnableSequence(
        extract_video_id_step,
        fetch_transcript_step,
        embeddings_step,
        store_step,
        format_step
    )
    return chain

def main():
    # Create the chain
    chain = create_transcript_embedding_chain()
    
    # Get YouTube URL from user
    youtube_url = input("Enter YouTube video URL: ")
    
    try:
        # Run the chain
        result = chain.invoke(youtube_url)
        
        # Print results
        print(f"Video ID: {result['video_id']}")
        print(f"Transcript excerpt: {result['transcript_excerpt']}")
        print(f"Vector Embeddings (first 10 values): {result['embeddings_first_10']}")
        print(f"Total embedding length: {result['embedding_length']}")
        print(f"ChromaDB Status: {result['chroma_status']}")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()