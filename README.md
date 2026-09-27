# Project IRA 🚀

Project IRA is a professional, decoupled AI Orchestrator designed for developers. It moves beyond simple chat interfaces by implementing a tiered architecture (API $\rightarrow$ Service $\rightarrow$ Data) and integrating real-time web browsing and multimodal capabilities.

## ✨ Key Features
- **Multimodal Brain**: Support for text and image inputs (Vision-capable).
- **Real-time Intelligence**: Integrated web searching and headless scraping to bypass the knowledge cutoff.
- **Session Management**: Persistent chat history stored in a local SQLite database.
- **Model Agnostic**: Designed to work with any Ollama-compatible model.
- **Liquid Glass UI**: A cosmic, minimalist frontend built with React and TypeScript.

## 🛠️ Architecture
The project follows a decoupled service pattern:
- **Interface Layer**: FastAPI endpoints for seamless communication.
- **Service Layer**: Business logic handling AI orchestration and tool loops.
- **Client/Data Layer**: Direct interactions with Ollama API and SQLite.

## 🚀 Quick Start (Linux Mint / Ubuntu)

### 1. Backend Setup
```bash
# Navigate to project root
cd project_IRA

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Start the server
python main.py
```

### 2. Frontend Setup
```bash
cd ira-ui
npm install
npm run dev
```

### 3. Model Requirement
Ensure you have Ollama installed and the desired model pulled:
```bash
ollama pull gemma4:31b-cloud # For text/reasoning
ollama pull llava           # For vision support
```

## 🛡️ License
Private Repository - For development and research purposes.
