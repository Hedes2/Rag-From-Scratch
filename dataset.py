import torch
from torch.utils.data import DataLoader,Dataset
import random
from transformers import AutoTokenizer

class CustomDataset(Dataset):

    def __init__(self,anchor_dict,positive_dict,qrels_dict,seq_len:int=15):

        self.anchor=anchor_dict
        self.seq_len=seq_len
        self.positive=positive_dict
        self.qrels=qrels_dict
        self.tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased")

#creating negatives
        self.all_positives_ids=list(self.positive.keys())



    def __len__(self):
        return len(self.qrels)



    def tokenize(self,text):

        encoded=self.tokenizer(
            text,
            max_length=self.seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"

        )

        return encoded["input_ids"].squeeze(dim=0)
    #squeeze removes a dim if size of that dim is 1



    def __getitem__(self,idx):
        item=self.qrels[idx]
        #item is a dcitonary in itself


        query_id=item['query-id']
        positive_ans_id=item['corpus-id']

        anchor=self.anchor[query_id]
        positive_ans=self.positive[positive_ans_id]
        
        random_id=random.choice(self.all_positives_ids)
        while random_id == positive_ans_id:
            random_id = random.choice(self.all_positives_ids)
        negative_ans=self.positive[random_id]

        anchor_tensor = self.tokenize(anchor)
        positive_tensor = self.tokenize(positive_ans)
        negative_tensor = self.tokenize(negative_ans)

        return {
            "anchor":anchor_tensor,
            "positive":positive_tensor,
            "negative":negative_tensor
        }
    '''{
    "anchor":   tensor([ 101, 2054, 2024, 2019, 3136, 1029,  102,    0,    0,    0,    0,    0,    0,    0,    0]),
    
    "positive": tensor([ 101, 1037, 3136, 2003, 1037, 2115, 2000, 1996, 3185, 1012,  102,    0,    0,    0,    0]),
    
    "negative": tensor([ 101, 4954, 2024, 2125, 3448, 1012,  102,    0,    0,    0,    0,    0,    0,    0,    0])
}'''


# --- RUN THIS AT THE BOTTOM OF YOUR FILE ---
if __name__ == "__main__":
    import json
    import csv

    print("1. Loading local data...")

    # Load the Positive/Negative paragraphs
    positive_dict = {}
    with open(r"D:\AI ML learnings\Rag implementation\corpus.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            positive_dict[item["_id"]] = item.get("text", "") 

    # Load the Anchor questions
    anchor_dict = {}
    with open(r"D:\AI ML learnings\Rag implementation\queries.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            anchor_dict[item["_id"]] = item.get("text", "")

    # Load the Answer Key using train.tsv!
    qrels_dict = []
    with open(r"D:\AI ML learnings\Rag implementation\qrels\train.tsv", "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader) # Skip the header row
        for row in reader:
            # row[0] is query-id, row[1] is corpus-id
            qrels_dict.append({'query-id': row[0], 'corpus-id': row[1]})

    print(f"Loaded {len(anchor_dict)} queries, {len(positive_dict)} paragraphs, and {len(qrels_dict)} pairs!")

    print("\n2. Initializing your CustomDataset...")
    my_dataset = CustomDataset(anchor_dict, positive_dict, qrels_dict, seq_len=15)

    print("3. Testing the Waiter (__getitem__)...")
    sample = my_dataset[0]
    
    print("\n--- TEST OUTPUT (TRIPLET 0 TENSORS) ---")
    print(f"ANCHOR TENSOR:   {sample['anchor']}")
    print(f"POSITIVE TENSOR: {sample['positive']}")
    print(f"NEGATIVE TENSOR: {sample['negative']}")
    print(f"Shape of one tensor: {sample['anchor'].shape}")