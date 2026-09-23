import os
import time
from pathlib import Path
from dotenv import load_dotenv
from tqdm.auto import tqdm
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load environment variables
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = "us-east-1"
PINECONE_INDEX_NAME = "medical-index"

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

UPLOAD_DIR = "./uploaded_pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
spec = ServerlessSpec(cloud="aws", region=PINECONE_ENV)
existing_indexes = [i["name"] for i in pc.list_indexes()]

if PINECONE_INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=768,  # For GoogleGenerativeAI embeddings
        metric="dotproduct",
        spec=spec
    )
    while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
        time.sleep(1)

index = pc.Index(PINECONE_INDEX_NAME)

from modules.embedding_utils import embed_documents_safe

# Load, split, embed and upsert PDF content
def load_vectorstore(uploaded_files):
    embed_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", output_dimensionality=768)
    file_paths = []

    for file in uploaded_files:
        save_path = Path(UPLOAD_DIR) / file.filename
        with open(save_path, "wb") as f:
            f.write(file.file.read())
        file_paths.append(str(save_path))

    for file_path in file_paths:
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(documents)

        texts = [chunk.page_content for chunk in chunks]
        metadatas = []
        for chunk in chunks:
            meta = dict(chunk.metadata) if chunk.metadata else {}
            meta["text"] = chunk.page_content
            metadatas.append(meta)

        ids = [f"{Path(file_path).stem}-{i}" for i in range(len(chunks))]

        print(f"🔍 Embedding {len(texts)} chunks for {file_path}...")
        embeddings = embed_documents_safe(
            embed_model=embed_model,
            texts=texts,
            batch_size=15,             # 15 chunks per batch to stay well under Gemini limits
            delay_between_batches=1.5, # 1.5s delay between batches (~40 requests/min max)
            max_retries=6              # Retries with exponential backoff on 429
        )

        print(f"📤 Uploading {len(embeddings)} vectors to Pinecone...")
        vectors = list(zip(ids, embeddings, metadatas))
        upsert_batch_size = 100
        with tqdm(total=len(vectors), desc="Upserting to Pinecone") as progress:
            for i in range(0, len(vectors), upsert_batch_size):
                batch_vectors = vectors[i:i + upsert_batch_size]
                index.upsert(vectors=batch_vectors)
                progress.update(len(batch_vectors))

        print(f"✅ Upload complete for {file_path}")

