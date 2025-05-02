

import streamlit as st
import fitz 
import os
import numpy as np
import hashlib
import pickle
import httpx
from typing import List
from openai import OpenAI
from chromadb import PersistentClient
from chromadb.api.types import Documents, Embeddings, EmbeddingFunction

CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "text-embedding-ada-002"
CHAT_MODEL = "gpt-3.5-turbo"
CHROMA_PATH = ".chroma"
COLLECTION_NAME = "pdf_chunks_openai"
CACHE_DIR = ".cache"

os.makedirs(CACHE_DIR, exist_ok=True)

class CustomHTTPClient(httpx.Client):
    def __init__(self, *args, **kwargs):
        kwargs.pop("proxies", None)
        super().__init__(*args, **kwargs)

client = OpenAI(http_client=CustomHTTPClient())

class OpenAIEmbeddingFunction(EmbeddingFunction[Documents]):
    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        BATCH_SIZE = 100
        for i in range(0, len(input), BATCH_SIZE):
            batch = input[i:i + BATCH_SIZE]
            response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
            embeddings.extend([e.embedding for e in response.data])
        return embeddings

def extract_text_from_pdf(path: str) -> str:
    doc = fitz.open(path)
    return "\n".join([page.get_text() for page in doc])

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size - overlap)]

def get_file_hash(filepaths: List[str]) -> str:
    m = hashlib.md5()
    for path in sorted(filepaths):
        with open(path, 'rb') as f:
            m.update(f.read())
    return m.hexdigest()

def generate_answer(context: str, query: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI assistant. Answer only using the provided context. "
                "If the answer is not in the context, say 'I don't know, I couldn't find the relevant answer'."
            )
        },
        {
            "role": "user",
            "content": f"""
Context:
{context}

Question: {query}
Answer:"""
        }
    ]
    response = client.chat.completions.create(model=CHAT_MODEL, messages=messages)
    return response.choices[0].message.content.strip()


st.set_page_config(page_title="RAG with OpenAI", layout="centered")
st.title("nRAG with OpenAI + ChromaDB")

query = st.text_input("Enter your query:")

if query:
    with st.spinner("Processing PDFs and retrieving relevant information..."):
        all_chunks = []
        file_paths = []

        for filename in os.listdir("data"):
            if filename.endswith(".pdf"):
                pdf_path = os.path.join("data", filename)
                file_paths.append(pdf_path)
                text = extract_text_from_pdf(pdf_path)
                chunks = chunk_text(text)
                all_chunks.extend(chunks)

        pdf_hash = get_file_hash(file_paths)
        cache_file = os.path.join(CACHE_DIR, f"{pdf_hash}.pkl")
        is_cache_hit = os.path.exists(cache_file)

        ids = [f"chunk_{i}" for i in range(len(all_chunks))]

        if is_cache_hit:
            with open(cache_file, "rb") as f:
                cached_ids, cached_chunks = pickle.load(f)
        else:
            st.info("Caching document chunks for future queries...")
            cached_ids = ids
            cached_chunks = all_chunks
            with open(cache_file, "wb") as f:
                pickle.dump((cached_ids, cached_chunks), f)

        chroma_client = PersistentClient(path=CHROMA_PATH)
        collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=OpenAIEmbeddingFunction())

        if len(collection.get()["ids"]) == 0:
            st.info("Adding new embeddings to ChromaDB...")
            collection.add(documents=cached_chunks, ids=cached_ids)
        elif is_cache_hit:
            st.info("Reusing embeddings from ChromaDB + cache.")

      
        results = collection.query(query_texts=[query], n_results=3)
        top_chunks = results['documents'][0]
        context = "\n".join(top_chunks)

        st.subheader(" Retrieved Context")
        st.text_area("Context passed to the model:", context, height=200)

        answer = generate_answer(context, query)
        if not answer or answer.strip() == ".":
            answer = "Sorry, I couldn't find a relevant answer in the documents."

    st.subheader("Answer:")
    st.write(answer)
