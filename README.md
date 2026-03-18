# Financial RAG Search Engine

A custom Retrieval-Augmented Generation (RAG) system specialized for financial data. 

**Note:** This project does not use pre-built sentence transformer APIs or wrappers. The underlying architecture—including Multi-Head Attention, Positional Encoding, and Transformer blocks—is built entirely from scratch in PyTorch.

## Tech Stack
* **Deep Learning:** PyTorch
* **LLM:** Ollama (Llama 3)
* **Vector DB:** Custom In-Memory PyTorch Tensor
* **Frontend:** Streamlit

## Architecture
* **Custom Encoder:** A local Sentence Transformer (6 layers, 8 heads, 512-dim) built with raw PyTorch layers.
* **Training Strategy:** Optimized using `TripletMarginLoss` with the FiQA dataset to learn financial semantic proximity.
* **Retrieval:** Document chunks are stored and searched sub-second via an in-memory tensor.
* **Generation:** Uses a locally hosted Llama 3 model to ensure complete data privacy.

## Getting Started

### Prerequisites
* Python 3.8+
* [Ollama](https://ollama.com/) installed

### Installation & Execution

```
ollama pull llama3

git clone [https://github.com/yourusername/financial-rag.git](https://github.com/yourusername/financial-rag.git)
cd financial-rag
pip install torch transformers streamlit PyPDF2 requests

streamlit run app.py
Repository Structure
Plaintext
.
├── app.py             # Streamlit UI
├── model.py           # Custom PyTorch RAGEncoder architecture
├── inference.py       # Retrieval and generation logic
├── ragpipeline.py     # PDF parsing and vector indexing
├── train.py           # Training pipeline
└── dataset.py         # FiQA data loading
Future Roadmap
[ ] Increase sequence length for broader context.

[ ] Implement context-aware chunking for financial reports.

[ ] Migrate to FAISS or ChromaDB for production vector storage.

[ ] Add CSV/Excel parsing support.
