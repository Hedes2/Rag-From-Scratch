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

```bash
# 1. Pull the LLM
ollama pull llama3

# 2. Clone and Install
git clone [https://github.com/yourusername/financial-rag.git](https://github.com/yourusername/financial-rag.git)
cd financial-rag
pip install torch transformers streamlit PyPDF2 requests

# 3. Run the App
streamlit run app.py



.
├── app.py             # Streamlit UI
├── model.py           # Custom PyTorch RAGEncoder architecture
├── inference.py       # Retrieval and generation logic
├── ragpipeline.py     # PDF parsing and vector indexing
├── train.py           # Training pipeline
└── dataset.py         # FiQA data loading
