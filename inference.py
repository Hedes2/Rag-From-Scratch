import torch
from transformers import AutoTokenizer
from model import RAGEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 15
VOCAB_SIZE = 30522

print("Loading Model and Database...")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = RAGEncoder(vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN).to(device)

# Load your custom brain
model.load_state_dict(torch.load("rag_encoder_trained.pth", map_location=device))
model.eval()

# Load the Database!
db_data = torch.load("finance_vector_db.pt", map_location=device)
vector_database = db_data["database_tensor"]
text_chunks = db_data["text_chunks"]
print(f"Loaded {len(text_chunks)} financial chunks into memory.")

import requests

def generate_answer(query, context_chunks):
    prompt = f"Context: {context_chunks}\n\nQuestion: {query}\nAnswer:"
    
    print("\n--- GENERATING ANSWER ---")
    response = requests.post("http://localhost:11434/api/generate", json={
        "model": "llama3", 
        "prompt": prompt,
        "stream": False
    })
    
    print("\n" + response.json()['response'] + "\n")

def search(query, top_k=3):
# Convert the query into a vector
    encoded = tokenizer(
        query, 
        max_length=SEQ_LEN, 
        padding="max_length", 
        truncation=True, 
        return_tensors="pt"
    )
    input_tensor = encoded["input_ids"].to(device)
    
    with torch.no_grad():
        query_vector = model(input_tensor) # Shape: [1, 512]
        
        #Calculate Euclidean Distance between query and ALL database vectors
        #p=2 means Euclidean distance (the exact metric we trained the model on!)
        distances = torch.cdist(query_vector, vector_database, p=2) # Shape: [1, N]
        
        # Get the Top-K closest chunks
        # largest=False because we want the SMALLEST distance between vectors
        top_distances, top_indices = torch.topk(distances, k=top_k, largest=False)
        
        # Print the retrieved context
        print("\n--- TOP MATCHES (RETRIEVAL) ---")
        retrieved_chunks = []
        for i, idx in enumerate(top_indices[0]):
            dist = top_distances[0][i].item()
            chunk_text = text_chunks[idx.item()]
            retrieved_chunks.append(chunk_text)
            print(f"{i+1}. [Distance: {dist:.2f}] {chunk_text}")
        
        # Generate the final answer!
        generate_answer(query, retrieved_chunks)

if __name__ == "__main__":
    print("\n" + "="*50)
    print("FINANCIAL RAG SEARCH ENGINE INITIALIZED (OLLAMA EDITION)")
    print("="*50)
    
    while True:
        user_query = input("\nAsk a finance question (or type 'quit'): ")
        if user_query.lower() == 'quit':
            break
        search(user_query)