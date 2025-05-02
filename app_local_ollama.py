import streamlit as st
import fitz 
import os
import requests
import hashlib
import pickle
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


st.set_page_config(page_title="Local RAG - Mistral Q4 via Ollama", layout="centered")
st.title(" Local RAG with Mistral Q4 via Ollama")


EMBEDDING_MODEL_NAME = "all-MiniLM-L12-v2"
embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)


query = st.text_input("Ask a question about the PDFs:")

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

def generate_answer_with_ollama(context, query):
    prompt = f"""You are an AI assistant. Use only the context provided below to answer the user's question.

If the answer is not present in the context, say "I don't know, I can't find the relevant answer".

Context:
{context}

Question: {query}

Answer:"""
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "mistral", "prompt": prompt, "stream": False}
    )
    return response.json()["response"].strip()

# Main logic
if query:
    with st.spinner(" Processing PDFs from 'data/' and searching..."):
        all_chunks = []
        file_paths = []

        for filename in os.listdir("data"):
            if filename.endswith(".pdf"):
                path = os.path.join("data", filename)
                file_paths.append(path)
                text = extract_text_from_pdf(path)
                all_chunks.extend(chunk_text(text))

        
        pdf_hash = get_pdf_hash(file_paths)
        cache_path = os.path.join(".cache", f"{pdf_hash}.pkl")
        os.makedirs(".cache", exist_ok=True)

        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                ids, cached_chunks = pickle.load(f)
        else:
            ids = [f"chunk_{i}" for i in range(len(all_chunks))]
            cached_chunks = all_chunks
            with open(cache_path, "wb") as f:
                pickle.dump((ids, cached_chunks), f)

     
        chroma_client = PersistentClient(path=".chroma")
        embed_fn = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)
        collection = chroma_client.get_or_create_collection(name="pdf_chunks_mistral_local", embedding_function=embed_fn)

        if len(collection.get()["ids"]) == 0:
            collection.add(documents=cached_chunks, ids=ids)
            st.info("Stored chunks in Chroma.")
        else:
            st.info("Chroma collection already populated.")


        results = collection.query(query_texts=[query], n_results=3)
        top_chunks = results['documents'][0]
        context = "\n".join(top_chunks)

        st.subheader(" Retrieved Context")
        st.text_area("Context passed to the model:", context, height=200)

        answer = generate_answer_with_ollama(context, query)
        if not answer or answer.strip() == ".":
            answer = "Sorry, I couldn't find a relevant answer in the documents."

    st.subheader(" Answer:")
    st.write(answer)
