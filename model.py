import torch 
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):

    def __init__(self,d_model:int,vocab_size:int):   
        super().__init__()
        self.vocab_size=vocab_size
        self.d_model=d_model
        self.embeddings=nn.Embedding(vocab_size,d_model)

    def forward(self,x):
        return self.embeddings(x)*math.sqrt(self.d_model)

class Postional_encodding(nn.Module):

    def __init__(self,seq_len:int,d_model:int):
        super().__init__()
        self.d_model=d_model 
        self.seq_len=seq_len

        pe=torch.zeros(seq_len,d_model)
        #[0,0,0,0]pe
        #[0,0,0,0]
        #[0,0,0,0]
        #[0,0,0,0]
        position=torch.arange(0,seq_len).reshape(-1,1)
        i=torch.arange(0,d_model,2).float()

        div_term=torch.exp(i*(-math.log(10000.0)/d_model))

        pe[:,0::2]=torch.sin(position*div_term)
        pe[:,1::2]=torch.cos(position*div_term)

        pe=pe.unsqueeze(0)

        self.register_buffer('pe',pe)#not a learnable parameter anymore


    def forward(self,x):
         x=x+self.pe[:,:x.shape[1],:]
         return x

class MultiHeadAttention(nn.Module):

    def __init__(self,d_model:int,h:int):
        super().__init__()
        self.h=h
        self.d_model=d_model
        self.d_k=d_model//h
        self.w_q=nn.Linear(d_model,d_model)
        self.w_k=nn.Linear(d_model,d_model)
        self.w_v=nn.Linear(d_model,d_model)

        self.w_o = nn.Linear(d_model, d_model)


    def forward(self,x):
        Q=self.w_q(x)
        K=self.w_k(x)
        V=self.w_v(x)
        #[15X512]

        batch_size,seq_len,_=Q.shape
        Q = Q.view(batch_size, seq_len, self.h, self.d_k).transpose(1,2)
        K = K.view(batch_size, seq_len, self.h, self.d_k).transpose(1,2)
        V = V.view(batch_size, seq_len, self.h, self.d_k).transpose(1,2)

        K=K.transpose(-2,-1)
        attention_scores=(Q@K)/math.sqrt(self.d_k)
        attention_scores=attention_scores.softmax(dim=-1)

        embeddings=attention_scores@V

#thoda kam smaj aaya
        embeddings = embeddings.transpose(1,2).contiguous()
        embeddings = embeddings.view(batch_size, seq_len, self.d_model)


        

        embeddings = self.w_o(embeddings)

        return embeddings
    


class Transformer(nn.Module):
    
    def __init__(self,d_model:int,h:int):
        super().__init__()
        self.attention=MultiHeadAttention(d_model,h)
        self.norm1=nn.LayerNorm(d_model)

        self.ffn=nn.Sequential(
            nn.Linear(d_model,4*d_model),
            nn.GELU(),
            nn.Linear(4*d_model,d_model)
        )

        self.norm2=nn.LayerNorm(d_model)

    def forward(self,x):

        output=self.attention(x)
        output=self.norm1(output+x)

        output2=self.ffn(output)
        output2=self.norm2(output2+output)


        return output2
    


class RAGEncoder(nn.Module):
    def __init__(self,vocab_size:int,d_model:int=512,seq_len:int=15,h:int=8,N:int=6):

        super().__init__()
        self.embed=InputEmbeddings(d_model,vocab_size)
        self.pos=Postional_encodding(seq_len,d_model)

        self.layers=nn.ModuleList([Transformer(d_model,h) for _ in range (N)])


    def forward(self,x):
        x=self.embed(x)
        x=self.pos(x)

        for layer in self.layers:
            x=layer(x)


        #mean pooling karege ab
        #because it will gerate 15 vector for every word but we need one only
        #dim=(batchxseq_lenx512)
        chunk_vector=x.mean(dim=1)

        return chunk_vector
    

# --- THE TEST BLOCK GOES AT THE VERY BOTTOM ---
if __name__ == "__main__":
    vocab_size = 10000 
    batch_size = 2      
    seq_len = 15        
    d_model = 512       

    model = RAGEncoder(vocab_size=vocab_size, d_model=d_model, seq_len=seq_len)
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))

    print(f"Input shape: {dummy_input.shape} --> (Batch Size: {batch_size}, Words: {seq_len})")
    output_vectors = model(dummy_input)
    print(f"Output shape: {output_vectors.shape} --> (Batch Size: {batch_size}, Vector Size: {d_model})")
    print("Success! Your engine works perfectly.")



    

        






