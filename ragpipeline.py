from PyPDF2 import PdfReader
import torch

from model import RAGEncoder

from transformers import AutoTokenizer

def extract_pdf(pdf_path):
    reader=PdfReader(pdf_path)

    text=""

    for pages in reader.pages:
        text+=pages.extract_text()+" "


    output=text.split()
    #convert to list

    return output



def chunkers(output,chunk_size=15):


    chunks=[]

    for i in range(0,len(output),chunk_size):
        chunk_words=output[i:i+chunk_size]

        if(len(chunk_words)==chunk_size):
            chunk_string=" ".join(chunk_words)
            chunks.append(chunk_string)
        

    return chunks 



if __name__ == "__main__":
    device=torch.device("cuda"if torch.cuda.is_available() else"cpu")
    my_pdf = r'D:\AI ML learnings\Rag implementation\hf-strategies.pdf'
    SEQ_LEN=15
    VOCAB_SIZE=30522

    words_list=extract_pdf(my_pdf)
    final_chunks=chunkers(words_list)
    print(f"no of chunks are {len(final_chunks)}")

    print("2. Loading the trained RAGEncoder...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = RAGEncoder(vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN).to(device)
    
    # Load the weights you just finished training!
    model.load_state_dict(torch.load("rag_encoder_trained.pth", map_location=device))
    model.eval() # Tell the model we are just testing, not training


    #now we will convert all chunks into vectors and save it
    all_vectors=[]

    with torch.no_grad():

        for chunk in final_chunks:
            # Tokenize exactly like you did in CustomDataset
            encoded = tokenizer(
                chunk, 
                max_length=SEQ_LEN, 
                padding="max_length", 
                truncation=True, 
                return_tensors="pt"
            )
            input_tensor = encoded["input_ids"].to(device)
            
            # Pass it through the Multi-Head Attention blocks
            chunk_vector = model(input_tensor) # Shape: [1, 512]
            all_vectors.append(chunk_vector)

    # Stack all those [1, 512] vectors into one giant [N, 512] tensor
    vector_database = torch.cat(all_vectors, dim=0)


    torch.save({
        "database_tensor": vector_database,
        "text_chunks": final_chunks
    }, "finance_vector_db.pt")
    
    print(f"\nSUCCESS! Saved Vector Database of shape: {vector_database.shape}")