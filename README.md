# 🔒 Secure LLM API with FastAPI & Ollama

![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-FF6F00?logo=ollama&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)

A production-ready API for secure access to local LLMs with:
- API key authentication
- Usage credits system
- Docker deployment
- CI/CD pipeline support

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- [Ollama installed](https://ollama.com/download)
- Mistral model: `ollama pull mistral`

### Installation
```bash
git clone https://github.com/yourusername/llm-api.git
cd llm-api

# Install dependencies
pip install -r requirements.txt

# Set up environment
echo "API_KEY=your_super_secret_key" > .env