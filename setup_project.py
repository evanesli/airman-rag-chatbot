"""
Project Structure Setup Script
Run this to create all necessary folders and files
"""
import os

def create_project_structure():
    """Creates the complete project folder structure"""
    
    # Define the structure
    structure = {
        'data': ['raw', 'processed', 'vector_store'],
        'src': ['ingestion', 'retrieval', 'generation', 'evaluation', 'api'],
        'tests': [],
        'notebooks': [],
        'config': [],
        'logs': []
    }
    
    # Create directories
    for main_dir, sub_dirs in structure.items():
        os.makedirs(main_dir, exist_ok=True)
        print(f"✅ Created: {main_dir}/")
        
        for sub_dir in sub_dirs:
            path = os.path.join(main_dir, sub_dir)
            os.makedirs(path, exist_ok=True)
            print(f"   ✅ Created: {path}/")
    
    # Create __init__.py files
    init_paths = [
        'src/__init__.py',
        'src/ingestion/__init__.py',
        'src/retrieval/__init__.py',
        'src/generation/__init__.py',
        'src/evaluation/__init__.py',
        'src/api/__init__.py',
        'tests/__init__.py'
    ]
    
    for init_path in init_paths:
        with open(init_path, 'w') as f:
            f.write('"""Package initialization"""\n')
        print(f"✅ Created: {init_path}")
    
    # Create placeholder files
    placeholder_files = [
        'src/ingestion/pdf_loader.py',
        'src/ingestion/chunking.py',
        'src/ingestion/embeddings.py',
        'src/retrieval/vector_search.py',
        'src/retrieval/reranker.py',
        'src/generation/llm_client.py',
        'src/generation/answer_generator.py',
        'src/evaluation/metrics.py',
        'src/evaluation/evaluator.py',
        'src/api/main.py'
    ]
    
    for file_path in placeholder_files:
        with open(file_path, 'w') as f:
            f.write(f'"""{os.path.basename(file_path)} - Implementation goes here"""\n')
        print(f"✅ Created: {file_path}")
    
    # Create config.yaml
    with open('config/config.yaml', 'w') as f:
        f.write("""# Configuration file for AIRMAN RAG Chatbot

# Chunking parameters
chunking:
  chunk_size: 1000
  chunk_overlap: 200
  strategy: "semantic"  # Options: semantic, fixed, sliding

# Embedding model
embeddings:
  model_name: "sentence-transformers/all-MiniLM-L6-v2"
  dimension: 384

# Vector store
vector_store:
  type: "faiss"
  index_type: "IndexFlatL2"
  
# LLM settings
llm:
  provider: "openai"  # Options: openai, anthropic, local
  model: "gpt-3.5-turbo"
  temperature: 0.1
  max_tokens: 1000

# Retrieval settings
retrieval:
  top_k: 5
  similarity_threshold: 0.7
  use_reranking: true

# API settings
api:
  host: "0.0.0.0"
  port: 8000
  debug: true
""")
    print("✅ Created: config/config.yaml")
    
    # Create .env.example
    with open('.env.example', 'w') as f:
        f.write("""# Environment Variables Template
# Copy this to .env and fill in your actual values

# OpenAI API Key (if using OpenAI)
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic API Key (if using Claude)
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Other API keys
HUGGINGFACE_API_KEY=your_huggingface_key_here

# Database (if using PostgreSQL)
DATABASE_URL=postgresql://user:password@localhost:5432/airman_rag

# Application settings
ENVIRONMENT=development
LOG_LEVEL=INFO
""")
    print("✅ Created: .env.example")
    
    # Create .gitignore
    with open('.gitignore', 'w') as f:
        f.write("""# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/

# Data files
data/raw/*.pdf
data/processed/
data/vector_store/
*.faiss
*.pkl

# Environment variables
.env

# IDE
.vscode/
.idea/
*.swp
*.swo

# Logs
logs/
*.log

# Jupyter
.ipynb_checkpoints/

# Models and embeddings
models/
*.bin
*.pt

# OS
.DS_Store
Thumbs.db

# Testing
.pytest_cache/
.coverage
htmlcov/

# Docker
*.tar
*.gz
""")
    print("✅ Created: .gitignore")
    
    # Create README.md template
    with open('README.md', 'w') as f:
        f.write("""# AIRMAN - Aviation Document RAG Chatbot

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
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
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
""")
    print("✅ Created: README.md")
    
    print("\n" + "="*60)
    print("✨ Project structure created successfully!")
    print("="*60)
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Copy .env.example to .env and add your API keys")
    print("3. Place PDF files in data/raw/")
    print("4. Start coding! 🚀")

if __name__ == "__main__":
    create_project_structure()