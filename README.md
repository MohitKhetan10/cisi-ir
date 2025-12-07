# CISI Information Retrieval System

This project implements an IR pipeline on the **CISI dataset**, comparing **TF-IDF** and **BM25** retrieval models.

## 📂 Structure
- `data/` → CISI dataset files
- `src/` → Modular Python code
- `results/` → Retrieval outputs
- `notebooks/` → Demo notebook (`cisi_ir_demo.ipynb`)

## 🚀 Features
- Parse CISI dataset
- Preprocess text
- TF-IDF retrieval
- BM25 retrieval
- Evaluation metrics: MAP, Precision@10, nDCG@10, MRR

## 📊 Example Results
| Model   | MAP   | Precision@10 | nDCG@10 | MRR   |
|---------|-------|--------------|---------|-------|
| TF‑IDF  | 0.165 | 0.323        | 0.352   | 0.576 |
| BM25    | 0.186 | 0.341        | 0.377   | 0.619 |

## ▶️ Usage
Run the demo notebook:
```bash
jupyter notebook notebooks/cisi_ir_demo.ipynb
