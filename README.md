🎙️ YouTube Transcript Chatbot
A Python-based chatbot that fetches transcripts from YouTube videos, embeds them using Hugging Face models, stores them in ChromaDB, and allows users to ask context-aware questions about the content. Powered by LangChain, transformers, and Azure AI for LLM-based answering.

🚀 Features
🎥 Transcript Extraction: Automatically extracts English transcripts from YouTube videos.

🧠 Embedding with Transformers: Converts transcript chunks into vector embeddings using sentence-transformers/all-MiniLM-L6-v2.

🧬 ChromaDB Integration: Stores transcript embeddings in an in-memory ChromaDB for fast similarity search.

🗣️ Conversational Q&A: Ask natural language questions about the video content.

🌐 FastAPI Backend: Offers RESTful endpoints for frontend integration or other services.

☁️ Azure OpenAI Integration: Uses Azure-hosted GPT-4o model to answer questions based on transcript context.

## 🖼️ Sample Interface

<p align="center">
  <img src="images/Podcast consumption" width="800"/>
</p>
<p align="center">
  <img src="images/Insight on a sub topic" width="800"/>
</p>
<p align="center">
  <img src="images/Generate notes" width="800"/>
</p>





🛠️ Project Structure
bash
Copy
Edit
📁 your-project/
├── chatbot_transcript.py        # Transcript processing and embedding chain
├── chatbot_query.py             # Querying and conversation logic
├── api_server.py                # FastAPI app to expose endpoints
├── .env                         # Stores Hugging Face and GitHub credentials
└── requirements.txt             # All Python dependencies
🧰 Requirements
Python 3.8+

Hugging Face Transformers

LangChain

ChromaDB

YouTube Transcript API

Azure AI SDK

FastAPI

🔐 Environment Variables
Create a .env file in the root directory with the following:

ini
Copy
Edit
HUGGINGFACE_API_KEY=your_huggingface_api_key
GITHUB_TOKEN=your_github_access_token_for_azure_models
📦 Installation
bash
Copy
Edit
# Clone the repo
git clone https://github.com/your-username/yt-transcript-chatbot.git
cd yt-transcript-chatbot

# Set up virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
🧪 Running the CLI Version
Run the terminal chatbot interface:

bash
Copy
Edit
python chatbot_query.py
Paste a YouTube URL when prompted.

Then ask questions like:

"What is the video about?"

"Summarize the first section."

"What are the main points discussed?"

Type exit to end the session.

🌐 Running the API Server
To launch the FastAPI backend:

bash
Copy
Edit
uvicorn api_server:app --reload
Endpoints
POST /upload-video-url
Uploads and processes a YouTube video.

Request Body:

json
Copy
Edit
{
  "url": "https://www.youtube.com/watch?v=abc123xyz"
}
POST /query
Ask a question based on the uploaded transcript.

Request Body:

json
Copy
Edit
{
  "query": "What does the speaker say about machine learning?"
}
GET /response
Returns the latest model response.

🧠 How It Works
Extract Video ID – From a full YouTube URL.

Fetch Transcript – Using youtube_transcript_api.

Split Transcript – Into overlapping chunks using a custom recursive splitter.

Embed Text – Convert text chunks into embeddings using Hugging Face Transformers.

Store in ChromaDB – Enables fast vector search for later queries.

Query Handling – User queries are embedded and matched to similar chunks.

LLM Answering – GPT-4o (via Azure) responds using the retrieved transcript chunks.

📈 Use Cases
Educational video summarization

Customer service video analysis

Podcast and lecture question answering

Content review for accessibility

📋 Future Improvements
Persistent storage (ChromaDB with file-backed DB)

Multi-language transcript support

Frontend UI integration (React/Next.js)

Upload support for custom audio/video files
