# 🎙️ Chatterbox TTS API

**Python 3.10+ · FastAPI · GPU Accelerated**  
**License:** Internal Use Only

A **production-ready, long-form Text-to-Speech (TTS) API** built with **FastAPI** and **Resemble AI's Chatterbox model**.

This service is optimized for **long content** such as stories, articles, books, and reports using **intelligent chunking** and **natural pauses** to deliver realistic, high-quality speech.

Designed strictly for **internal company / office use**.

---

## 🚀 Features

- 📝 **Long-text optimized**  
  Smart sentence-based chunking for fast and stable processing

- 🎧 **Natural prosody**  
  Configurable silence between chunks for realistic speech flow

- 🔊 **Fixed premium voice**  
  Single predefined high-quality voice for consistent output

- 🔐 **Simple & secure API**  
  Only text input required

- ❤️ **Health check endpoint**  
  Production monitoring ready

- ⚡ **GPU acceleration**  
  Automatic CUDA detection

- 📊 **Professional logging**

- 📘 **Interactive API docs**  
  Swagger UI & ReDoc enabled

---

## ⚡ Quick Start

### 1️⃣ Clone Repository
```bash
git clone https://github.com/Khairul-islam99/ENGLISH_TTS.git
cd ENGLISH_TTS
```
2️⃣ Create .env File
```bash
DEFAULT_VOICE=C:\path\to\your\high_quality_voice_sample.mp3
```
3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
4️⃣ Run Server
```bash
python main.py
```
Or using Uvicorn:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
### 🌐 Access

📄 API Docs (Swagger)
http://localhost:8000/docs

❤️ Health Check
http://localhost:8000/health


