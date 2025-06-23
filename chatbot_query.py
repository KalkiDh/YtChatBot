import os
import re
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain_core.runnables import RunnableLambda, RunnableSequence
from transformers import AutoTokenizer, AutoModel
import torch
import chromadb
from chromadb.utils import embedding_functions
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage, AssistantMessage
from azure.core.credentials import AzureKeyCredential

# Load environment variables
load_dotenv()
hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
github_token = os.getenv("GITHUB_TOKEN")

if not hf_api_key:
    raise ValueError("HUGGINGFACE_API_KEY not found in .env file")
if not github_token:
    raise ValueError("GITHUB_TOKEN not found in .env file")

# Initialize in-memory ChromaDB client (volatile)
chroma_client = chromadb.Client()

# Initialize conversation history
conversation_history = []

# Custom Recursive Text Splitter
class CustomRecursiveTextSplitter:
    def __init__(self, chunk_size=500, chunk_overlap=50, separators=None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or [
            "\n\n", "\n", " ", ".", ",", "\u200b", "\uff0c", "\u3001", "\uff0e", "\u3002", ""
        ]

    def split_text(self, text):
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
                            overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                            current_chunk = current_chunk[overlap_start:] + (separator if current_chunk else "") + split
                        else:
                            sub_chunks = self.split_text(split)
                            chunks.extend(sub_chunks)
                            current_chunk = ""
                if current_chunk:
                    chunks.append(current_chunk)
                return chunks if chunks else []

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

# Function to split transcript
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
    
    try:
        chroma_client.delete_collection(collection_name)
    except:
        pass
    
    collection = chroma_client.create_collection(
        name=collection_name,
        embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    )
    
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
            "chunk_text": chunk[:1000]
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
        "chroma_status": f"Stored {len(data['transcript_chunks'])} transcript chunks for video {data['video_id']} in volatile ChromaDB"
    }

# Function to format embedding output
def format_embedding_output(data: dict) -> dict:
    return {
        "video_id": data["video_id"],
        "transcript_chunk_count": len(data["transcript_chunks"]),
        "first_chunk_excerpt": data["transcript_chunks"][0][:100] + "..." if data["transcript_chunks"] and len(data["transcript_chunks"][0]) > 100 else data["transcript_chunks"][0] if data["transcript_chunks"] else "",
        "first_chunk_embeddings": data["embeddings"][0][:10] if data["embeddings"] else [],
        "embedding_length": len(data["embeddings"][0]) if data["embeddings"] else 0,
        "chroma_status": data["chroma_status"]
    }

# Function to embed user query
def embed_query(query: str) -> dict:
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_api_key)
    model = AutoModel.from_pretrained(model_name, token=hf_api_key)
    
    inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    query_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy().tolist()
    
    return {"query": query, "query_embedding": query_embedding}

# Function to retrieve similar chunks
def retrieve_similar_chunks(data: dict) -> dict:
    collection_name = "youtube_transcripts"
    try:
        collection = chroma_client.get_collection(collection_name)
        results = collection.query(
            query_embeddings=[data["query_embedding"]],
            n_results=3
        )
        similar_chunks = [
            {
                "chunk_text": doc,
                "metadata": meta
            } for doc, meta in zip(results["documents"][0], results["metadatas"][0])
        ]
        return {
            "query": data["query"],
            "query_embedding": data["query_embedding"],
            "similar_chunks": similar_chunks
        }
    except Exception as e:
        return {
            "query": data["query"],
            "query_embedding": data["query_embedding"],
            "similar_chunks": [],
            "error": f"Error retrieving chunks: {str(e)}"
        }

# Function to create prompt
def create_prompt(data: dict) -> dict:
    query = data["query"]
    similar_chunks = data.get("similar_chunks", [])
    
    context = ""
    for idx, chunk in enumerate(similar_chunks):
        context += f"Chunk {idx + 1} (Video ID: {chunk['metadata']['video_id']}, Chunk Index: {chunk['metadata']['chunk_index']}):\n{chunk['chunk_text']}\n\n"
    
    system_prompt = (
        "You are a helpful assistant that answers queries based on relevant transcript chunks from a YouTube video. "
        "Use the provided context to inform your response. If no relevant chunks are available, answer to the best of your knowledge. "
        f"Context:\n{context if context else 'No relevant transcript chunks found.'}"
    )
    
    conversation_history.append({"role": "user", "content": UserMessage(query)})
    
    messages = [
        SystemMessage(system_prompt),
        *[msg["content"] for msg in conversation_history]
    ]
    
    return {
        "query": query,
        "similar_chunks": similar_chunks,
        "messages": messages
    }

# Function to query Azure AI model
def query_azure_ai(data: dict) -> dict:
    endpoint = "https://models.github.ai/inference"
    model = "openai/gpt-4.1"
    
    client = ChatCompletionsClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(github_token)
    )
    
    try:
        response = client.complete(
            messages=data["messages"],
            temperature=1.0,
            top_p=1.0,
            model=model
        )
        answer = response.choices[0].message.content
        
        conversation_history.append({"role": "assistant", "content": AssistantMessage(answer)})
        
        return {
            "query": data["query"],
            "similar_chunks": data["similar_chunks"],
            "answer": answer,
            "conversation_history": conversation_history
        }
    except Exception as e:
        return {
            "query": data["query"],
            "similar_chunks": data["similar_chunks"],
            "answer": f"Error querying model: {str(e)}",
            "conversation_history": conversation_history
        }

# Function to format query output
def format_query_output(data: dict) -> dict:
    return {
        "query": data["query"],
        "answer": data["answer"],
        "similar_chunks": [
            {
                "video_id": chunk["metadata"]["video_id"],
                "chunk_index": chunk["metadata"]["chunk_index"],
                "chunk_text": chunk["chunk_text"][:100] + "..." if len(chunk["chunk_text"]) > 100 else chunk["chunk_text"]
            } for chunk in data["similar_chunks"]
        ],
        "conversation_history": [
            {"role": msg["role"], "content": msg["content"].content}
            for msg in data["conversation_history"]
        ]
    }

# Define vector embedding chain
def create_embedding_chain():
    return RunnableSequence(
        RunnableLambda(extract_video_id),
        RunnableLambda(fetch_transcript),
        RunnableLambda(split_transcript),
        RunnableLambda(text_to_embeddings),
        RunnableLambda(store_in_chromadb),
        RunnableLambda(format_embedding_output)
    )

# Define query chain
def create_query_chain():
    return RunnableSequence(
        RunnableLambda(embed_query),
        RunnableLambda(retrieve_similar_chunks),
        RunnableLambda(create_prompt),
        RunnableLambda(query_azure_ai),
        RunnableLambda(format_query_output)
    )

def main():
    print("Welcome to the YouTube Transcript Chatbot!")
    
    # Get YouTube video URL
    youtube_url = input("Enter YouTube video URL: ")
    
    # Run vector embedding chain
    embedding_chain = create_embedding_chain()
    try:
        embedding_result = embedding_chain.invoke(youtube_url)
        print(f"\nVideo ID: {embedding_result['video_id']}")
        print(f"Number of Transcript Chunks: {embedding_result['transcript_chunk_count']}")
        print(f"First Chunk Excerpt: {embedding_result['first_chunk_excerpt']}")
        print(f"First Chunk Embeddings (first 10 values): {embedding_result['first_chunk_embeddings']}")
        print(f"Embedding Length: {embedding_result['embedding_length']}")
        print(f"ChromaDB Status: {embedding_result['chroma_status']}")
        print("\nNow you can ask questions about the video. Type 'exit' to quit.\n")
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        return
    
    # Run query chain
    query_chain = create_query_chain()
    while True:
        query = input("Enter your query: ")
        if query.lower() == "exit":
            print("Exiting chatbot.")
            break
        
        try:
            result = query_chain.invoke(query)
            print(f"\nQuery: {result['query']}")
            print(f"Answer: {result['answer']}")
            print("\nSimilar Transcript Chunks:")
            for chunk in result["similar_chunks"]:
                print(f"- Video ID: {chunk['video_id']}, Chunk Index: {chunk['chunk_index']}")
                print(f"  Text: {chunk['chunk_text']}")
            print("\nConversation History:")
            for msg in result["conversation_history"]:
                print(f"{msg['role'].capitalize()}: {msg['content']}")
            print("\n" + "="*50 + "\n")
        except Exception as e:
            print(f"Error processing query: {str(e)}")

if __name__ == "__main__":
    main()