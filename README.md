# Financial RAG Search Engine (Built from Scratch)

A custom Retrieval-Augmented Generation (RAG) system specialized for financial data. 

> [!IMPORTANT]
> **Zero APIs for Core Logic**: This project does NOT use pre-built sentence transformer APIs or high-level wrappers. The **Sentence Transformer architecture was built entirely from scratch** using PyTorch, including the implementation of Multi-Head Attention, Positional Encoding, and Transformer blocks. This custom model was then used to power a complete RAG pipeline.

## 🏗️ Architecture

- **Custom Sentence Transformer**: A from-scratch `RAGEncoder` (6 layers, 8 heads, 512-dim) built using raw PyTorch layers.
- **Dense Retrieval**: No external embedding services are used. Semantic representations are generated locally by the custom-built encoder.
- **Training Strategy**: Uses `TripletMarginLoss` with the FiQA financial dataset to learn semantic proximity.
- **Vector Database**: Chunks are embedded and stored in an in-memory PyTorch tensor for sub-second retrieval.
- **Local LLM Integration**: Uses Ollama with the `llama3` model for privacy and performance.

## 🚀 Getting Started

### Prerequisites

1.  **Python 3.8+**
2.  **Ollama**: Install from [ollama.com](https://ollama.com) and pull the Llama 3 model:
    ```bash
    ollama pull llama3
    ```
3.  **Dependencies**:
    ```bash
    pip install torch transformers streamlit PyPDF2 requests
    ```

### Running the App

1.  Start the Streamlit interface:
    ```bash
    streamlit run app.py
    ```
2.  Upload a financial PDF through the UI.
3.  Ask questions about the uploaded content!

## 📁 Repository Structure

- `app.py`: Streamlit-based user interface.
- `model.py`: Neural architecture definitions for the RAG encoder.
- `inference.py`: Retrieval and Ollama generation logic.
- `ragpipeline.py`: Offline scripts for PDF processing and vector database indexing.
- `train.py` & `dataset.py`: Training pipeline and data loading.

## 🔧 Future Improvements

- Increasing sequence length for more context.
- Implementing more advanced chunking strategies.
- Swapping in-memory search with FAISS or ChromaDB for scalability.
