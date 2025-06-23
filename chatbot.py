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

# Initialize in-memory ChromaDB client (volatile, resets every run)
chroma_client = chromadb.Client()  # In-memory client, no persistence

# Custom Recursive Text Splitter
class CustomRecursiveTextSplitter:
    def __init__(self, chunk_size=500, chunk_overlap=50, separators=None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or [
            "\n\n", "\n", " ", ".", ",", "\u200b", "\uff0c", "\u3001", "\uff0e", "\u3002", ""
        ]

    def split_text(self, text):
        """Recursively split text into chunks based on separators."""
        if len(text) <= self.chunk_size:
            return [text.strip()] if text.strip() else []

        chunks = []
        for separator in self.separators:
            if separator in text:
                splits = text.split(separator)
                current_chunk = ""
                for split in splits:
                    split = split.strip()
                    if not split:
                        continue
                    temp_chunk = current_chunk + (separator if current_chunk else "") + split
                    if len(temp_chunk) <= self.chunk_size:
                        current_chunk = temp_chunk
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                            # Add overlap from the end of the current chunk
                            overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                            current_chunk = current_chunk[overlap_start:] + (separator if current_chunk else "") + split
                        else:
                            # If split is too large, recurse on it
                            sub_chunks = self.split_text(split)
                            chunks.extend(sub_chunks)
                            current_chunk = ""
                if current_chunk:
                    chunks.append(current_chunk)
                return chunks if chunks else []

        # If no separators work, split by character
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk = text[i:i + self.chunk_size]
            chunks.append(chunk)
            if len(chunk) < self.chunk_size:
                break
        return [chunk.strip() for chunk in chunks if chunk.strip()]

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

# Function to split transcript using CustomRecursiveTextSplitter
def split_transcript(data: dict) -> dict:
    text_splitter = CustomRecursiveTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ".", ",", "\u200b", "\uff0c", "\u3001", "\uff0e", "\u3002", ""]
    )
    chunks = text_splitter.split_text(data["transcript"])
    return {
        "video_id": data["video_id"],
        "transcript_chunks": chunks
    }

# Function to convert text chunks to embeddings
def text_to_embeddings(data: dict) -> dict:
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_api_key)
    model = AutoModel.from_pretrained(model_name, token=hf_api_key)
    
    embeddings = []
    for chunk in data["transcript_chunks"]:
        inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        chunk_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy().tolist()
        embeddings.append(chunk_embedding)
    
    return {
        "video_id": data["video_id"],
        "transcript_chunks": data["transcript_chunks"],
        "embeddings": embeddings
    }

# Function to store embeddings in ChromaDB
def store_in_chromadb(data: dict) -> dict:
    collection_name = "youtube_transcripts"
    
    # Reset collection to ensure a fresh start each run
    try:
        chroma_client.delete_collection(collection_name)
    except:
        pass  # Collection may not exist yet
    
    # Create new collection
    collection = chroma_client.create_collection(
        name=collection_name,
        embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    )
    
    # Store each chunk's embedding
    ids = []
    embeddings = []
    metadatas = []
    documents = []
    for idx, (chunk, embedding) in enumerate(zip(data["transcript_chunks"], data["embeddings"])):
        doc_id = f"transcript_{data['video_id']}_chunk_{idx}"
        ids.append(doc_id)
        embeddings.append(embedding)
        metadatas.append({
            "video_id": data["video_id"],
            "chunk_index": idx,
            "chunk_text": chunk[:1000]  # Limit metadata size
        })
        documents.append(chunk)
    
    collection.add(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=documents
    )
    
    return {
        "video_id": data["video_id"],
        "transcript_chunks": data["transcript_chunks"],
        "embeddings": data["embeddings"],
        "chroma_status": f"Stored {len(data['transcript_chunks'])} data chunks for video {data['video_id']} in volatile ChromaDB"
    }

# Function to format output
def format_output(data: dict) -> dict:
    return {
        "video_id": data["video_id"],
        "transcript_chunk_count": len(data["transcript_chunks"]),
        "first_chunk_excerpt": data["transcript_chunks"][0][:100] + "..." if data["transcript_chunks"] and len(data["transcript_chunks"][0]) > 100 else data["transcript_chunks"][0] if data["transcript_chunks"] else "",
        "first_chunk_embeddings": data["embeddings"][0][:10] if data["embeddings"] else [],
        "embedding_length": len(data["embeddings"][0]) if data["embeddings"] else 0,
        "chroma_status": data["chroma_status"]
    }

# Define LangChain chain
def create_transcript_chain():
    # Step 1: Extract video ID
    extract_video_id_step = RunnableLambda(extract_video_id)
    
    # Step 2: Fetch transcript
    fetch_transcript_step = RunnableLambda(fetch_transcript)
    
    # Step 3: Split transcript into chunks
    split_transcript_step = RunnableLambda(split_transcript)
    
    # Step 4: Convert chunks to embeddings
    embeddings_step = RunnableLambda(text_to_embeddings)
    
    # Step 5: Store in ChromaDB
    store_step = RunnableLambda(store_in_chromadb)
    
    # Step 6: Format output
    format_step = RunnableLambda(format_output)
    
    # Create the chain
    chain = RunnableSequence(
        extract_video_id_step,
        fetch_transcript_step,
        split_transcript_step,
        embeddings_step,
        store_step,
        format_step
    )
    return chain

def main():
    # Create the chain
    chain = create_transcript_chain()
    
    # Get YouTube URL from user
    youtube_url = input("Enter YouTube video URL: ")
    
    try:
        # Run the chain
        result = chain.invoke(youtube_url)
        
        # Print results
        print(f"Video ID: {result['video_id']}")
        print(f"Number of Transcript Chunks: {result['transcript_chunk_count']}")
        print(f"First Chunk Excerpt: {result['first_chunk_excerpt']}")
        print(f"First Chunk Embeddings (first 10 values): {result['first_chunk_embeddings']}")
        print(f"Embedding Length: {result['embedding_length']}")
        print(f"ChromaDB Status: {result['chroma_status']}")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()