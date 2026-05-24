# ⚖️ Legal Prior Case Retrieval using Knowledge Graphs and RAG

![Python](https://img.shields.io/badge/Python-3.x-blue)
![NLP](https://img.shields.io/badge/NLP-Legal%20AI-green)
![RAG](https://img.shields.io/badge/RAG-Retrieval%20Augmented-orange)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-red)
![License](https://img.shields.io/badge/License-Academic-lightgrey)

A legal information retrieval system that compares Knowledge Graph (KG), Retrieval-Augmented Generation (RAG), and hybrid retrieval approaches for identifying relevant prior legal cases from large legal document collections.

The project focuses on improving legal case retrieval by combining semantic similarity, graph-based reasoning, and neural re-ranking techniques.

---

# 📌 Project Description

Legal research heavily depends on identifying relevant prior cases and precedents. Traditional keyword-based systems often fail to capture semantic relationships and contextual meaning within legal documents.

This project implements multiple retrieval pipelines using:
- Knowledge Graphs (KG)
- Retrieval-Augmented Generation (RAG)
- RA-KG-LLM Hybrid Retrieval
- Neural Re-ranking
- Reciprocal Rank Fusion (RRF)

The system compares structural retrieval methods with semantic retrieval methods and evaluates their effectiveness on legal prior case retrieval tasks.

---

## 📌 Key Highlights

- Hybrid Legal Retrieval System
- Knowledge Graph + RAG Integration
- Neural Re-ranking Pipeline
- FAISS-based Semantic Search
- Legal NLP Workflow

# ✨ Features

- Legal prior case retrieval system
- Knowledge Graph construction using legal entities
- Semantic retrieval using embeddings and FAISS
- Retrieval-Augmented Generation (RAG) pipeline
- Neural cross-encoder re-ranking
- Hybrid retrieval using Reciprocal Rank Fusion (RRF)
- Comparative evaluation of retrieval approaches
- Legal NLP workflow for precedent analysis

---

# 🛠️ Tech Stack

## Languages
- Python

## Libraries & Tools
- NumPy
- Pandas
- FAISS
- Scikit-learn
- TensorFlow
- NLP Techniques

## Concepts Used
- Knowledge Graphs
- Retrieval-Augmented Generation (RAG)
- Dense Embeddings
- TF-IDF
- Neural Re-ranking
- Information Retrieval
- Legal NLP

---

# 📂 Dataset Information

Dataset Used:
- IL-TUR Prior Case Retrieval Dataset

Dataset contains:
- Legal case documents
- Prior case references
- Judicial text data
- Legal entities such as statutes, judges, and courts

Full dataset is not included due to GitHub file size limitations.
---

# 🔄 Project Architecture / Workflow

The system combines structural graph relationships and semantic retrieval techniques.

## Workflow

1. Input legal query case
2. Preprocess legal text
3. Extract legal entities
4. Generate TF-IDF and dense embeddings
5. Construct Knowledge Graph
6. Retrieve candidate cases using KG and RAG
7. Apply neural re-ranking
8. Combine results using Reciprocal Rank Fusion (RRF)
9. Return top relevant prior cases

---

# 🧠 Model / Methodology Used

## 1. Knowledge Graph (KG)
- Nodes represent legal cases and entities
- Edges represent relationships such as:
  - Shared statutes
  - Shared judges
  - Shared courts

## 2. Retrieval-Augmented Generation (RAG)
- Semantic retrieval using:
  - TF-IDF similarity
  - Dense embeddings
  - FAISS vector indexing

## 3. RA-KG-LLM
- Combines:
  - Graph-based reasoning
  - Semantic retrieval
  - Contextual understanding

## 4. Neural Re-ranking
- Cross-encoder model assigns relevance scores
- Improves top-k retrieval quality

## 5. Hybrid Fusion
- Uses Reciprocal Rank Fusion (RRF)
- Combines outputs from:
  - KG
  - RAG
  - RA-KG-LLM

---

# ⚙️ Installation Steps

## Clone Repository

```bash
git clone https://github.com/yourusername/legal-prior-case-retrieval.git

cd legal-prior-case-retrieval
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Usage Instructions

## Run Jupyter Notebook

```bash
jupyter notebook
```

Run the notebook step-by-step:
1. Data preprocessing
2. Knowledge graph construction
3. Embedding generation
4. Retrieval pipeline
5. Neural re-ranking
6. Evaluation

---

# 📊 Results / Output

## Evaluation Metrics
- MAP
- Precision
- Recall
- F1-score
- nDCG
- Hits@K

## Performance Comparison

| Method | F1@10 | nDCG@10 |
|--------|--------|----------|
| KG | 0.087 | 0.109 |
| RAG-BM25 | 0.266 | 0.347 |
| RAG+LLM | 0.232 | 0.297 |
| RA-KG-LLM | 0.127 | 0.148 |
| Hybrid-RRF | 0.254 | 0.334 |

## Key Findings
- RAG-based methods outperform KG-only retrieval
- Semantic retrieval performs better than structural retrieval
- Hybrid methods provide balanced performance
- Neural re-ranking improves retrieval quality

---

# 📁 Folder Structure

```bash
legal-prior-case-retrieval/
│
├── dataset/
├── notebooks/
├── models/
├── screenshots/
├── report/
├── requirements.txt
├── main.ipynb
└── README.md
```

---

# 🚀 Future Improvements

- Legal-BERT integration
- GraphRAG implementation
- Improved LLM-based re-ranking
- Real-time deployment
- Better graph-enhanced retrieval systems

---

# 📸 Screenshots

## Workflow Diagram
Add workflow diagram here

```md
![Workflow](screenshots/workflow.png)
```

## Output Screenshots
Add retrieval results and evaluation graphs here

```md
![Results](screenshots/results.png)
```

---

# ✅ Conclusion

This project presents a comparative analysis of Knowledge Graph, Retrieval-Augmented Generation (RAG), and hybrid retrieval approaches for legal prior case retrieval.

The results show that semantic retrieval methods outperform purely structural methods, while hybrid approaches provide balanced and reliable performance for legal information retrieval tasks.

---

# 👨‍💻 Author Details

## Dinesh G
B.Tech Artificial Intelligence  
National Institute of Technology Karnataka (NITK), Surathkal

- GitHub: (https://github.com/Dinesh090420)
- LinkedIn: www.linkedin.com/in/garbhapu-dinesh-140bb6373

---

# 📜 License

This project is developed for academic and research purposes.