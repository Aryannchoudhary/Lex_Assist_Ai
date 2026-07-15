# Lex-Assist: AI Legal Copilot & Risk Analyzer

Lex-Assist is an enterprise-grade, full-stack AI legal assistant designed to automate contract analysis, identify high-risk legal clauses, and provide a context-aware Retrieval-Augmented Generation (RAG) conversational engine over dense legal documents.

By combining custom **Deep Learning (BiLSTM)** for clause classification with state-of-the-art **Large Language Models (Qwen 2.5)** and **Vector Databases (ChromaDB)**, Lex-Assist transforms raw, complex PDFs into structured, actionable risk insights with statistical visual analytics.

---

## 🚀 Key Features

* **AI-Powered Risk Assessment:** Automatically scans uploaded PDFs to detect and flag high-risk legal clauses (e.g., Indemnity, Limitation of Liability, Termination) using a custom deep learning network.
* **Interactive Legal Copilot (RAG):** Allows users to ask natural language questions about their contracts. Backed by ChromaDB and Qwen 2.5 (7B) to guarantee context-anchored, anti-hallucination answers.
* **Executive Data Analytics:** Features a built-in statistics dashboard that parses, counts, and visualizes contract risk profiles using Pandas, Matplotlib, and Seaborn.
* **Microservices Architecture:** Fully containerized backend (FastAPI) and frontend (Streamlit) services running seamlessly via Docker.
* **Automated CI/CD Pipeline:** Includes a configured GitHub Actions workflow (`deploy.yml`) for instant, continuous integration and automated staging.

---

## 🛠️ The Tech Stack

| Domain | Technologies Used |
| --- | --- |
| **Core Engine** | Python |
| **Deep Learning** | TensorFlow, Keras, BiLSTM, GloVe Word Embeddings |
| **Data Processing** | Pandas, NumPy, Regular Expressions |
| **Vector DB & NLP** | ChromaDB, LangChain, Hugging Face Transformers |
| **LLM Orchestration** | Qwen 2.5 (7B), Advanced Prompt Engineering |
| **Data Visualization** | Matplotlib, Seaborn |
| **API & Backend** | FastAPI, Uvicorn, Docker |
| **Frontend UI** | Streamlit |
| **DevOps & CI/CD** | Docker Compose, GitHub Actions |

---

## 📂 Project Architecture & Directory Structure

```text
LEX_ASSIST_AI/
│
├── .github/workflows/
│   └── deploy.yml              # Automated GitHub Actions CI/CD pipeline
│
├── backend/
│   ├── app/
│   │   ├── api/                # API endpoints (Upload, Query, Health)
│   │   ├── db/                 # Vector DB setups (ChromaDB)
│   │   ├── models/             # Custom ML models & pipeline loading logic
│   │   ├── services/           # PDF parsing, text cleaning, LLM RAG pipelines
│   │   ├── __init__.py
│   │   └── main.py             # FastAPI entrypoint
│   ├── .dockerignore
│   ├── Dockerfile              # Docker configuration for FastAPI Backend
│   └── requirements.txt
│
├── frontend/
│   ├── app.py                  # Streamlit Web Application
│   ├── Dockerfile              # Docker configuration for Streamlit Frontend
│   └── requirements.txt
│
├── glove/                      # Directory for GloVe pre-trained vectors
│   └── glove.6B.100d.txt
│
├── ml_training/
│   ├── data/                   # Raw & processed training datasets
│   │   └── master_clauses.csv
│   ├── notebooks/
│   │   └── 01_data_exploration.ipynb # Model design, evaluation & visuals
│   ├── saved_models/
│   │   └── bilstm_clause_pipeline.pkl # Saved weights and tokenizer artifacts
│   └── train_classifier.py     # Clean executable script to retrain the BiLSTM
│
├── .gitignore                  # Keeps sensitive data & heavy cache off GitHub
├── docker-compose.yml          # Local multi-container orchestration
└── README.md

```

---

## 🧠 Machine Learning & Data Pipeline

### 1. Data Cleaning & Feature Engineering

Legal texts are plagued by nested arrays, carriage returns, and encoding anomalies. Lex-Assist runs clean pipelines to prepare raw texts:

* Normalizes special characters, whitespace, and lowercases raw text inputs.
* Uses **Stanford's pre-trained GloVe 100D word embeddings** to seed vocabulary maps.
* Tokenizes, pads, and truncates sequences to a uniform length of $250$ tokens for optimal memory footprint.

### 2. Neural Network Architecture

The classification engine utilizes a custom **Bidirectional Long Short-Term Memory (BiLSTM)** network built using Keras and TensorFlow:

* **Embedding Layer:** Initialized with frozen weights from the GloVe matrix to preserve pre-trained semantic relations.
* **Spatial Dropout:** Applied early ($0.2$) to prevent over-fitting on specific legal keywords.
* **Bidirectional LSTM (64 units):** Captures textual dependencies in both forward and backward directions (essential for complex legal prose).
* **Global Max Pooling & Dropout ($0.4$):** Reduces spatial dimensions and ensures high regularization.
* **Softmax Output Layer:** Delivers probabilistic risk classification across target legal clauses.

---

## ⚙️ Local Installation & Setup

### Prerequisites

Make sure you have [Docker](https://www.docker.com/) and [Git](https://git-scm.com/) installed on your machine.

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/lex-assist.git
cd lex-assist

```

### 2. Set Up Environment Variables

Create a `.env` file in the root directory:

```env
# Path to your vector database storage
CHROMA_DB_PATH=./backend/app/chroma_db

# API configuration (if utilizing external LLM models like Hugging Face or OpenAI)
HF_API_KEY=your_huggingface_api_token

```

### 3. Spin Up Docker Compose

You can boot both the FastAPI backend and Streamlit frontend containers with a single command:

```bash
docker-compose up --build

```

Once built and running:

* **Frontend UI:** Go to `http://localhost:8501`
* **Backend API Docs:** Go to `http://localhost:8000/docs`

---

## 📈 Executive Analytics & UI Dashboard

To make legal risk assessment actionable, the system computes and plots statistical insights of parsed contracts directly onto the UI:

1. **Clause Distribution Chart:** Highlights how often specific classes of clauses appear in historical documents using Seaborn barplots.
2. **Word Count Density Plots:** Displays text sequence distributions relative to the `MAX_SEQUENCE_LENGTH` cutoff.
3. **Training History Engine:** Visualizes training accuracy/loss curves to ensure continuous optimization and evaluate convergence.

---

## 🔄 Automated CI/CD (GitHub Actions)

The repository comes packaged with a continuous deployment workflow located in `.github/workflows/deploy.yml`.

Whenever a change is pushed to the `main` branch, GitHub Actions spins up an Ubuntu runner, securely grabs the encrypted deployment webhook from your repository secrets, and triggers an automated rebuild of your containers on your live hosting server.