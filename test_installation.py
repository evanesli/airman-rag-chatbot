"""Test if all packages are installed correctly"""

def test_imports():
    print("Testing imports...")
    
    try:
        import fastapi
        print("✅ FastAPI installed")
    except ImportError:
        print("❌ FastAPI not installed")
    
    try:
        import sentence_transformers
        print("✅ Sentence Transformers installed")
    except ImportError:
        print("❌ Sentence Transformers not installed")
    
    try:
        import faiss
        print("✅ FAISS installed")
    except ImportError:
        print("❌ FAISS not installed")
    
    try:
        import pypdf
        print("✅ PyPDF installed")
    except ImportError:
        print("❌ PyPDF not installed")
    
    try:
        import torch
        print("✅ PyTorch installed")
    except ImportError:
        print("❌ PyTorch not installed")
    
    try:
        from dotenv import load_dotenv
        print("✅ python-dotenv installed")
    except ImportError:
        print("❌ python-dotenv not installed")
    
    print("\n" + "="*50)
    print("Installation test complete!")
    print("="*50)

def test_api_key():
    from dotenv import load_dotenv
    import os
    
    load_dotenv()
    
    print("\nTesting API keys...")
    
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    
    if openai_key and openai_key != "your_openai_api_key_here":
        print("✅ OpenAI API key found")
    elif anthropic_key and anthropic_key != "your_anthropic_api_key_here":
        print("✅ Anthropic API key found")
    else:
        print("⚠️  No API key found - please add to .env file")

if __name__ == "__main__":
    test_imports()
    test_api_key()