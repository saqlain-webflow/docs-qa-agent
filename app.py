import streamlit as st
import os
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader

load_dotenv()

st.set_page_config(page_title="Ally - Support Assistant", page_icon="🤖")
st.title("🤖 Ally — Your Support Assistant")
st.caption("Prepared by Saqlain Ghazi")

def read_txt(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def read_pdf(filepath):
    reader = PdfReader(filepath)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def load_documents(folder="."):
    documents = []
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        if not os.path.isfile(filepath):
            continue
        if filename.endswith(".txt"):
            documents.append({"filename": filename, "content": read_txt(filepath)})
        elif filename.endswith(".pdf"):
            documents.append({"filename": filename, "content": read_pdf(filepath)})
    return documents

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(path="chroma_db")
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = client.get_or_create_collection(
        name="my_documents",
        embedding_function=embedding_fn
    )

    if collection.count() == 0:
        docs = load_documents(".")
        chunk_id = 0
        for doc in docs:
            chunks = chunk_text(doc['content'])
            for chunk in chunks:
                collection.add(
                    documents=[chunk],
                    metadatas=[{"source": doc['filename']}],
                    ids=[f"chunk_{chunk_id}"]
                )
                chunk_id += 1

    return collection

def search_chunks(collection, query, n_results=3):
    results = collection.query(query_texts=[query], n_results=n_results)
    return results['documents'][0]

def ask_ai(question, context_chunks):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    context = "\n\n".join(context_chunks)
    prompt = f"""You are Ally, a helpful customer support assistant. Answer the question based only on the context below. If the answer isn't in the context, say so politely.

Context:
{context}

Question: {question}

Answer:"""
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Load database once
collection = get_collection()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show old messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Input box
question = st.chat_input("Type your question here...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            chunks = search_chunks(collection, question)
            answer = ask_ai(question, chunks)
            st.write(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})