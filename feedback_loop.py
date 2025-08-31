# feedback_loop.py

import time
import json
import logging
import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from ranking_model import RankingModel  # import your model
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)


# -------------------------------
# Mock dataset for feedback data
# -------------------------------
class FeedbackDataset(Dataset):
    def __init__(self, feedback_data, product_embeddings, query_embeddings):
        """
        feedback_data: list of dicts with keys {query_id, product_id, relevance_score}
        product_embeddings: dict {product_id: embedding tensor}
        query_embeddings: dict {query_id: embedding tensor}
        """
        self.data = feedback_data
        self.product_embeddings = product_embeddings
        self.query_embeddings = query_embeddings

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        query_emb = self.query_embeddings[item["query_id"]]
        product_emb = self.product_embeddings[item["product_id"]]
        relevance = torch.tensor(item["relevance_score"], dtype=torch.float)
        return query_emb, product_emb, relevance


# -------------------------------
# Feedback loop trainer
# -------------------------------
class FeedbackTrainer:
    def __init__(self, model_name="bert-base-uncased", learning_rate=1e-4):
        self.embedding_model = SentenceTransformer(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = RankingModel().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

    def generate_embeddings(self, queries, products):
        query_embeddings = {
            qid: torch.tensor(self.embedding_model.encode(text)).float().to(self.device)
            for qid, text in queries.items()
        }
        product_embeddings = {
            pid: torch.tensor(self.embedding_model.encode(text)).float().to(self.device)
            for pid, text in products.items()
        }
        return query_embeddings, product_embeddings

    def train(self, feedback_data, query_embeddings, product_embeddings, epochs=3, batch_size=16):
        dataset = FeedbackDataset(feedback_data, product_embeddings, query_embeddings)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            total_loss = 0
            for query_emb, product_emb, relevance in dataloader:
                query_emb, product_emb, relevance = (
                    query_emb.to(self.device),
                    product_emb.to(self.device),
                    relevance.to(self.device),
                )

                self.optimizer.zero_grad()
                scores = self.model(query_emb, product_emb)
                loss = self.loss_fn(scores.squeeze(), relevance)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        return self.model

    def log_model(self, model, run_name="feedback_loop_run"):
        with mlflow.start_run(run_name=run_name):
            mlflow.pytorch.log_model(model, "ranking_model")
            logging.info("Model logged to MLflow")


# -------------------------------
# Example Feedback Loop Run
# -------------------------------
if __name__ == "__main__":
    logging.info("Starting feedback loop...")

    # Example data (in real case, this comes from user interactions)
    queries = {
        "q1": "wireless headphones",
        "q2": "gaming laptop",
    }

    products = {
        "p1": "Sony Wireless Noise Cancelling Headphones",
        "p2": "Bose Bluetooth Headphones",
        "p3": "Dell Gaming Laptop 16GB RAM",
        "p4": "Asus ROG Gaming Laptop with RTX 4060",
    }

    feedback_data = [
        {"query_id": "q1", "product_id": "p1", "relevance_score": 5.0},  # clicked & purchased
        {"query_id": "q1", "product_id": "p2", "relevance_score": 3.0},  # clicked only
        {"query_id": "q2", "product_id": "p3", "relevance_score": 4.0},  # long dwell time
        {"query_id": "q2", "product_id": "p4", "relevance_score": 5.0},  # clicked & purchased
    ]

    trainer = FeedbackTrainer()
    query_embeddings, product_embeddings = trainer.generate_embeddings(queries, products)
    model = trainer.train(feedback_data, query_embeddings, product_embeddings, epochs=5)
    trainer.log_model(model)

    logging.info("Feedback loop completed successfully.")
