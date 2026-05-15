# 🎮 GameTune AI

**GameTune AI** is a production-grade, single-file multimodal AI gaming assistant. It combines conversational intelligence, vision-based layout analysis, hardware optimization, and Retrieval-Augmented Generation (RAG) to provide a comprehensive coaching experience for competitive gamers.

---

## 🌟 Key Features

### 1. 💬 Chat Coach (RAG-Powered)
- **Multi-turn Memory**: Remembers your device, game, and playstyle.
- **RAG Integration**: Retrieves real-time strategies and tips from a local knowledge base (`knowledge_base/`).
- **Pro Advice**: Technical and concise coaching for titles like BGMI, Valorant, and COD Mobile.

### 2. 🖼️ Layout Analyzer (Multimodal Vision)
- **Screenshot Analysis**: Upload your control layout/HUD screenshot.
- **Ergonomic Suggestions**: Get advice on button placement for 2-finger, 3-finger (claw), or 4-finger setups.
- **HUD Optimization**: Identify inefficient button clusters or joystick sizes.

### 3. ⚡ Device Optimizer (Structured Analysis)
- **Spec-Based Recommendations**: Input your device (Mobile or Laptop) to get tailored graphics and FPS settings.
- **Thermal & Battery Tips**: Recommendations focused on maintaining stable performance during long sessions.
- **JSON Output**: Returns precise optimization profiles.

### 4. 📚 Pro Tips Explorer
- Search specifically through the indexed knowledge base for tactics and optimization guides.

---

## 🛠️ Tech Stack

- **UI Framework**: [Gradio](https://gradio.app/) (v6.x)
- **LLM Provider**: [OpenRouter API](https://openrouter.ai/)
  - **Main Model**: DeepSeek V4 Flash (Free)
  - **Vision Model**: Nemotron Nano 12B VL (Free)
  - **Reasoning Model**: Nemotron 3 Super 120B (Free)
- **Vector Database**: [ChromaDB](https://www.trychroma.com/)
- **Embeddings**: `all-MiniLM-L6-v2` (Sentence-Transformers)
- **Image Processing**: OpenCV & Pillow
- **Data Validation**: Pydantic

---

## 🚀 Getting Started

### 📋 Prerequisites
- Python 3.11+
- An OpenRouter API Key

### ⚙️ Installation

1. **Clone or Download** the project files.
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Environment Configuration**:
   Create a `.env` file in the root directory and add your API key:
   ```env
   OPENROUTER_API_KEY=your_openrouter_api_key_here
   ```

### 🏃 Running the Application
```bash
python main.py
```
The application will launch a local server (usually at `http://127.0.0.1:7860`) and provide a public share link compatible with Google Colab.

---

## 📁 Project Structure
The project is designed to be highly portable and modular within a single file:
- `main.py`: Core logic, RAG engine, Vision utilities, and Gradio UI.
- `requirements.txt`: Project dependencies.
- `knowledge_base/`: Directory containing `.txt` files for RAG indexing.
- `chroma_db/`: Persistent vector storage.
- `.env`: Secure API key management.

---

## 🔒 Security & Guardrails
- **Anti-Cheat Policy**: Refuses to provide assistance for hacks, cheats, or exploits.
- **Persona Integrity**: Maintains a professional "GameTune Coach" persona and protects system instructions.
- **Safe Outputs**: Mitigates prompt injection attempts through robust system prompting.

---

## 📝 License
This project is provided for educational and personal gaming optimization purposes.
