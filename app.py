import streamlit as st
import torch
import requests
from transformers import AutoTokenizer
from model import RAGEncoder 
from PyPDF2 import PdfReader

#made with ollama
# --- CONFIGURATION ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 15
VOCAB_SIZE = 30522

# --- CACHING FOR SPEED ---
@st.cache_resource
def load_models_and_db():
    print("Loading Model and Database...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = RAGEncoder(vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN).to(device)
    model.load_state_dict(torch.load("rag_encoder_trained.pth", map_location=device))
    model.eval()

    db_data = torch.load("finance_vector_db.pt", map_location=device)
    return tokenizer, model, db_data["database_tensor"], db_data["text_chunks"]

tokenizer, model, vector_database, text_chunks = load_models_and_db()

# --- UPLOAD AND PROCESS LOGIC ---
def process_pdf(uploaded_file):
    global vector_database, text_chunks
    
    # 1. Read PDF
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + " "
    
    # 2. Chunk it (using logic from ragpipeline.py)
    output = text.split()
    new_chunks = []
    chunk_size = 15
    for i in range(0, len(output), chunk_size):
        chunk_words = output[i:i+chunk_size]
        if len(chunk_words) == chunk_size:
            new_chunks.append(" ".join(chunk_words))
            
    # 3. Vectorize new chunks
    all_vectors = []
    with torch.no_grad():
        for chunk in new_chunks:
            encoded = tokenizer(
                chunk, max_length=SEQ_LEN, padding="max_length", 
                truncation=True, return_tensors="pt"
            )
            input_tensor = encoded["input_ids"].to(device)
            chunk_vector = model(input_tensor)
            all_vectors.append(chunk_vector)
            
    # 4. Add to existing database in memory
    new_vector_tensor = torch.cat(all_vectors, dim=0)
    vector_database = torch.cat([vector_database, new_vector_tensor], dim=0)
    text_chunks.extend(new_chunks)
    
    return len(new_chunks)

# --- RETRIEVAL LOGIC ---
def retrieve_chunks(query, top_k=3):
    encoded = tokenizer(
        query, max_length=SEQ_LEN, padding="max_length", 
        truncation=True, return_tensors="pt"
    )
    input_tensor = encoded["input_ids"].to(device)
    
    with torch.no_grad():
        query_vector = model(input_tensor)
        distances = torch.cdist(query_vector, vector_database, p=2) 
        top_distances, top_indices = torch.topk(distances, k=top_k, largest=False)
        
        results = []
        for i, idx in enumerate(top_indices[0]):
            results.append({
                "text": text_chunks[idx.item()],
                "distance": top_distances[0][i].item()
            })
        return results

# --- GENERATION LOGIC ---
def generate_llm_response(query, contexts):
    context_str = "\n- ".join([c["text"] for c in contexts])
    prompt = f"Context: {context_str}\n\nQuestion: {query}\nAnswer:"
    
    response = requests.post("http://localhost:11434/api/generate", json={
        "model": "llama3", 
        "prompt": prompt,
        "stream": False
    })
    return response.json()['response']

# --- UI LAYOUT ---
st.title("Financial Search Engine")

# File Upload Section
uploaded_file = st.file_uploader("Upload a new financial PDF", type="pdf")
if uploaded_file is not None:
    if st.button("Process Document"):
        st.write("Processing PDF and updating database...")
        num_chunks = process_pdf(uploaded_file)
        st.success(f"Successfully added {num_chunks} new chunks to the database!")

st.write("---")

# Search Section
user_query = st.text_input("Enter your question:")

if st.button("Search") and user_query:
    # Get chunks
    chunks = retrieve_chunks(user_query)
    
    # Get answer
    st.write("Generating answer...")
    answer = generate_llm_response(user_query, chunks)
    
    # Display everything
    st.header("Answer")
    st.write(answer)
    
    st.header("Retrieved Chunks")
    for i, c in enumerate(chunks):
        st.write(f"Chunk {i+1} (Distance: {c['distance']:.2f})")
        st.write(c['text'])
        st.write("---")
