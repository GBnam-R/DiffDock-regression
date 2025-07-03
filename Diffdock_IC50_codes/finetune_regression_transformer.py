import os
import math
import argparse
from typing import List, Dict, Any

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import XLNetModel

from terminator.tokenization import ExpressionBertTokenizer


class DockingDataset(Dataset):
    """Dataset wrapping DiffDock embeddings and pIC50 values."""

    def __init__(self, records: List[Dict[str, Any]], tokenizer: ExpressionBertTokenizer):
        self.records = records
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        tokenized = self.tokenizer(rec["compound"])
        emb_list = []
        if rec.get("protein_emb") is not None:
            emb_list.append(torch.tensor(rec["protein_emb"], dtype=torch.float))
        if rec.get("complex_emb") is not None:
            emb_list.append(torch.tensor(rec["complex_emb"], dtype=torch.float))
        features = torch.cat([e.flatten() for e in emb_list]) if emb_list else torch.empty(0)
        label = torch.tensor(float(rec["pIC50"]), dtype=torch.float)
        return tokenized, features, label


class Collator:
    def __init__(self, tokenizer: ExpressionBertTokenizer, feat_dim: int):
        self.tokenizer = tokenizer
        self.feat_dim = feat_dim

    def __call__(self, batch):
        tokens = [b[0] for b in batch]
        feats = [b[1] for b in batch]
        labels = torch.stack([b[2] for b in batch])

        enc = self.tokenizer.pad(tokens, return_tensors="pt")
        feat_batch = torch.zeros(len(feats), self.feat_dim)
        for i, f in enumerate(feats):
            feat_batch[i, : f.numel()] = f
        return enc, feat_batch, labels


class RegressionTransformerWithEmbeddings(nn.Module):
    def __init__(self, base_model: str, feature_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.transformer = XLNetModel.from_pretrained(base_model)
        hdim = self.transformer.config.hidden_size
        self.proj = nn.Linear(feature_dim, hdim)
        self.out = nn.Sequential(nn.Dropout(0.1), nn.Linear(hdim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, input_ids, attention_mask, features):
        out = self.transformer(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        pooled = out.mean(dim=1)
        proj = self.proj(features)
        x = torch.cat([pooled, proj], dim=1)
        return self.out(x).squeeze(-1)


def load_dataset(path: str, tokenizer: ExpressionBertTokenizer) -> DockingDataset:
    records = torch.load(path)
    return DockingDataset(records, tokenizer)


def train(args: argparse.Namespace) -> None:
    tokenizer = ExpressionBertTokenizer.from_pretrained(args.tokenizer)
    dataset = load_dataset(args.dataset, tokenizer)
    feat_dim = dataset[0][1].numel()

    collator = Collator(tokenizer, feat_dim)
    val_size = max(1, int(0.2 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RegressionTransformerWithEmbeddings(args.model, feat_dim, args.hidden_dim).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    best_rmse = math.inf
    os.makedirs(args.output, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        for enc, feats, labels in train_loader:
            enc = {k: v.to(device) for k, v in enc.items()}
            feats = feats.to(device)
            labels = labels.to(device)
            pred = model(**enc, features=feats)
            loss = loss_fn(pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        preds, labs = [], []
        with torch.no_grad():
            for enc, feats, labels in val_loader:
                enc = {k: v.to(device) for k, v in enc.items()}
                feats = feats.to(device)
                pred = model(**enc, features=feats)
                preds.extend(pred.cpu().tolist())
                labs.extend(labels.tolist())
        rmse = math.sqrt(sum((p - l) ** 2 for p, l in zip(preds, labs)) / len(labs))
        if rmse < best_rmse:
            best_rmse = rmse
            torch.save(model.state_dict(), os.path.join(args.output, "best_model.pt"))
        print(f"Epoch {epoch+1}/{args.epochs} RMSE: {rmse:.4f} (best {best_rmse:.4f})")


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune Regression Transformer on DiffDock dataset")
    parser.add_argument("dataset", help="Path to dataset file created with prepare_regression_dataset.py")
    parser.add_argument("output", help="Directory to store checkpoints")
    parser.add_argument("--model", default="xlnet-base-cased", help="Base transformer model")
    parser.add_argument("--tokenizer", default="vocabs/smallmolecules.txt", help="Tokenizer path")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    train(args)
