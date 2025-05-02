import streamlit as st
import os
import fitz
import requests
import hashlib
import pickle
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import httpx
from openai import OpenAI


CHROMA_PATH = ".chroma"
CACHE_DIR = ".cache"
COLLECTION_NAME = "pdf_chunks_combined"
EMBEDDING_MODEL_NAME = "all-MiniLM-L12-v2"
os.makedirs(CACHE_DIR, exist_ok=True)

st.set_page_config(page_title="RAG with Model Switch", layout="centered")
st.title("RAG App with OpenAI / Mistral ")


embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)


def extract_text_from_pdf(path):
    doc = fitz.open(path)
    return "\n".join([page.get_text() for page in doc])

def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size - overlap)]

def get_pdf_hash(filepaths):
    m = hashlib.md5()
    for path in sorted(filepaths):
        with open(path, 'rb') as f:
            m.update(f.read())
    return m.hexdigest()

def ask_openai(query, context):
    client = OpenAI(http_client=httpx.Client())
    messages = [
        {"role": "system", "content": "Answer using only the provided context. If the question is not in the context say 'I don't know, I couldn't find the relevant answer'"},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"}
    ]
    response = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
    return response.choices[0].message.content.strip()

def ask_ollama(query, context, model="mistral"):
    prompt = f"""You are a helpful assistant. Use only the context provided below to answer the user's question.

If the answer is not present in the context, say "I don't know, I couldn't find the relevant answer".

Context:
{context}

Question: {query}

Answer:"""
    response = requests.post("http://localhost:11434/api/generate", json={
        "model": model,
        "prompt": prompt,
        "stream": False
    })
    return response.json()["response"].strip()

model_choice = st.selectbox("Choose Model", ["openai", "mistral"])
query = st.text_input("Ask your question about the PDFs:")

if query:
    with st.spinner(" Processing PDFs and querying..."):
        all_chunks = []
        file_paths = [os.path.join("data", f) for f in os.listdir("data") if f.endswith(".pdf")]

        for path in file_paths:
            text = extract_text_from_pdf(path)
            all_chunks.extend(chunk_text(text))

        pdf_hash = get_pdf_hash(file_paths)
        cache_path = os.path.join(CACHE_DIR, f"{pdf_hash}.pkl")

        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                ids, cached_chunks = pickle.load(f)
        else:
            ids = [f"chunk_{i}" for i in range(len(all_chunks))]
            cached_chunks = all_chunks
            with open(cache_path, "wb") as f:
                pickle.dump((ids, cached_chunks), f)

        chroma_client = PersistentClient(path=CHROMA_PATH)
        embed_fn = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)
        collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=embed_fn)

        if len(collection.get()["ids"]) == 0:
            collection.add(documents=cached_chunks, ids=ids)

        results = collection.query(query_texts=[query], n_results=3, include=["documents", "distances"])
        top_chunks = results["documents"][0]
        scores = results["distances"][0]

        st.subheader(" Retrieved Context Chunks:")
        context = ""
        for i, (chunk, score) in enumerate(zip(top_chunks, scores), 1):
            with st.expander(f"Chunk {i} (Score: {score:.4f})"):
                st.write(chunk)
                context += chunk + "\n\n"

        if model_choice == "openai":
            answer = ask_openai(query, context)
        else:
            answer = ask_ollama(query, context, model=model_choice)

        if not answer.strip() or answer.strip() == ".":
            answer = "Sorry, I couldn't find a relevant answer in the documents."

        st.subheader("Answer")
        st.write(answer)
