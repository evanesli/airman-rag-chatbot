# AIRMAN - Aviation Document RAG Chatbot

An AI-powered chatbot that answers questions strictly from aviation documents (PPL/CPL/ATPL textbooks, SOPs, manuals) with zero hallucinations.

## 🚀 Features

- ✅ Document-grounded answers only
- ✅ Citation with page numbers
- ✅ Hallucination detection and prevention
- ✅ Confidence scoring
- ✅ Aviation-specific chunking strategy
- ✅ Multi-stage retrieval with reranking

## 📋 Requirements

- Python 3.10+
- FastAPI
- FAISS
- sentence-transformers
- OpenAI/Anthropic API key

## 🛠️ Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd airman-rag-chatbot
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your API keys
```

## 📚 Usage

### 1. Ingest Documents
```bash
python -m src.ingestion.pdf_loader --input data/raw/
```

### 2. Start API Server
```bash
uvicorn src.api.main:app --reload
```

### 3. Ask Questions
Visit http://localhost:8000/docs for interactive API documentation

## 📊 Evaluation

Run evaluation:
```bash
python -m src.evaluation.evaluator
```

## 🏗️ Architecture

[Add architecture diagram here]

## 📝 License

MIT License

## 👥 Author

Your Name - AIRMAN AI/ML Intern Technical Assignment
