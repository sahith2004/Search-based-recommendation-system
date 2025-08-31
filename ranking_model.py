# ranking_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import BertTokenizer, BertModel, AdamW

# --------------------------
# Dataset
# --------------------------
class RankingDataset(Dataset):
    def __init__(self, queries, products, labels, tokenizer, max_len=128):
        self.queries = queries
        self.products = products
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        q, p, label = self.queries[idx], self.products[idx], self.labels[idx]
        enc = self.tokenizer(
            q,
            p,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }

# --------------------------
# Ranking Model
# --------------------------
class RankingModel(pl.LightningModule):
    def __init__(self, model_name="bert-base-uncased", num_labels=2, lr=2e-5):
        super().__init__()
        self.save_hyperparameters()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.lr = lr

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.classifier(x)
        return logits

    def training_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["attention_mask"])
        loss = F.cross_entropy(logits, batch["label"])
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["attention_mask"])
        loss = F.cross_entropy(logits, batch["label"])
        preds = torch.argmax(logits, dim=1)
        acc = (preds == batch["label"]).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr)

# --------------------------
# Example Training Script
# --------------------------
if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    # Dummy dataset
    queries = ["wireless headphones", "gaming laptop", "phone case", "dslr camera"]
    products = [
        "Bluetooth wireless headphones with mic",
        "High performance gaming laptop RTX 3060",
        "Silicone phone case for iPhone",
        "Canon DSLR camera with 18-55mm lens"
    ]
    labels = [1, 1, 1, 1]  # Relevant = 1 (dummy data)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    q_train, q_val, p_train, p_val, y_train, y_val = train_test_split(
        queries, products, labels, test_size=0.2, random_state=42
    )

    train_ds = RankingDataset(q_train, p_train, y_train, tokenizer)
    val_ds = RankingDataset(q_val, p_val, y_val, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=2)

    model = RankingModel()

    trainer = pl.Trainer(max_epochs=3, accelerator="auto")
    trainer.fit(model, train_loader, val_loader)
