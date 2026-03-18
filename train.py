import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from model import RAGEncoder
from dataset import CustomDataset

import json
import csv

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE=16
LEARNING_RATE=2e-5
EPOCHS=3
MARGIN=1.0
VOCAB_SIZE=30522
SEQ_LEN=15

def load_fiqa_data():
    print("Loading data for training...")
    # Paths to your local files
    corpus_path = r"D:\AI ML learnings\Rag implementation\corpus.jsonl"
    queries_path = r"D:\AI ML learnings\Rag implementation\queries.jsonl"
    train_path = r"D:\AI ML learnings\Rag implementation\qrels\train.tsv"

    positive_dict = {}
    with open(r"D:\AI ML learnings\Rag implementation\corpus.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            positive_dict[item["_id"]] = item.get("text", "")

    anchor_dict = {}
    with open(r"D:\AI ML learnings\Rag implementation\queries.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            anchor_dict[item["_id"]] = item.get("text", "")

    qrels_train = []
    with open(r"D:\AI ML learnings\Rag implementation\qrels\train.tsv", "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader) 
        for row in reader:
            qrels_train.append({'query-id': row[0], 'corpus-id': row[1]})
    
    return anchor_dict, positive_dict, qrels_train


if __name__ == "__main__":
# 2. LOAD DATA AND MODEL
    anchors, positives, qrels = load_fiqa_data()

    train_dataset = CustomDataset(anchors, positives, qrels, seq_len=SEQ_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    #initialising the models , loss and optimizer
    model=RAGEncoder(vocab_size=VOCAB_SIZE).to(device)
    criterion=nn.TripletMarginLoss(margin=MARGIN,p=2)
    optimizer=optim.AdamW(model.parameters(),lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch tensors to GPU/CPU
            a = batch['anchor'].to(device)
            p = batch['positive'].to(device)
            n = batch['negative'].to(device)

            # FORWARD: Get 512-dim vectors for all three
            v_a = model(a)
            v_p = model(p)
            v_n = model(n)

            # LOSS: Calculate how well the model separated them
            loss = criterion(v_a, v_p, v_n)

            # BACKWARD: Update the weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)
        print(f"--- Epoch {epoch+1} Done. Average Loss: {avg_loss:.4f} ---")

    # 5. SAVE YOUR TRAINED MODEL
    torch.save(model.state_dict(), "rag_encoder_trained.pth")
    print("Model saved as rag_encoder_trained.pth")







