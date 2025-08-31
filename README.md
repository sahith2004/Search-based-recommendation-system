# Amazon Product Search Recommendation System

This project builds an **Amazon Product Search–based recommendation system** leveraging **SentenceTransformers + FAISS** for retrieval and a **BERT-based ranking model** for improved relevance. It integrates **Airflow** for pipeline orchestration, **MLflow** for experiment tracking, and implements a **feedback loop** for continuous optimization.  

---

##  Features
- **Dense Retrieval**: SentenceTransformers + FAISS for semantic product retrieval.  
- **Ranking Model**: BERT-based model to re-rank retrieved products by query relevance.  
- **Pipeline Orchestration**: Airflow DAGs for data ingestion, preprocessing, training, and evaluation.  
- **Experiment Tracking**: MLflow for logging metrics, parameters, and models.  
- **Feedback Loop**: User interactions (clicks, purchases) update embeddings and retrain the ranking model for adaptive improvements.  

---
## Metrics  

To evaluate the **retrieval** and **ranking** quality, the following metrics are used:  

### 1. **NDCG@K (Normalized Discounted Cumulative Gain)**  
- Measures ranking quality by considering both **position** and **relevance**.  
- High NDCG means relevant products are ranked higher.  
- Example: If a highly relevant product appears at rank 1, it contributes more than if it appeared at rank 5.  

### 2. **MRR (Mean Reciprocal Rank)**  
- Focuses on the **position of the first relevant item**.  
- Useful in search tasks where the top-ranked product should ideally be relevant.  
- Example: If the first relevant item is at position 2, score = 1/2.  

### 3. **Precision@K**  
- Measures the fraction of relevant items among the top-K retrieved items.  
- Example: If top-5 results contain 3 relevant products → Precision@5 = 0.6.  

### 4. **Recall@K**  
- Measures how many of the **total relevant items** were retrieved in the top-K.  
- Example: If there are 10 relevant products overall and top-5 contains 4 of them → Recall@5 = 0.4.  
