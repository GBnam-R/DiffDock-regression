import os
import math
import argparse

"""Fine-tune a Regression Transformer on DiffDock docking data.

This script loads a pretrained Regression Transformer model and fine tunes it on
a dataset created with ``create_regression_transformer_input.py``. The
implementation is inspired by ``scripts/run_language_modeling.py``. Earlier
versions added a small regression head on top of the model, but that head has
now been removed so that training follows a standard language modelling
objective.
"""

from typing import Any, Dict, List

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import XLNetLMHeadModel, get_linear_schedule_with_warmup

from terminator.tokenization import ExpressionBertTokenizer


class DockingDataset(Dataset):
    """Dataset providing token ids and embeddings."""

    def __init__(self, records: List[Dict[str, Any]]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        ids = torch.tensor(rec["input_ids"], dtype=torch.long)
        cemb = torch.tensor(rec["complex_emb"], dtype=torch.float32)
        pemb = torch.tensor(rec["protein_emb"], dtype=torch.float32)
        label = torch.tensor(float(rec["pIC50"]), dtype=torch.float32)
        return ids, cemb, pemb, label


class Collator:
    def __init__(self, tokenizer: ExpressionBertTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        ids = [b[0] for b in batch]
        cemb = torch.stack([b[1] for b in batch])
        pemb = torch.stack([b[2] for b in batch])
        labels = torch.stack([b[3] for b in batch])

        enc = self.tokenizer.pad({"input_ids": ids}, return_tensors="pt")
        return enc, cemb, pemb, labels


class RegressionTransformer(nn.Module):
    """XLNetLMHeadModel with embedding injection for complex and protein tokens."""

    def __init__(self, model_path: str, tokenizer: ExpressionBertTokenizer):
        super().__init__()
        # Load pretrained weights. ``model_path`` should contain a
        # ``pytorch_model.bin`` file as provided with the QED model.
        self.transformer = XLNetLMHeadModel.from_pretrained(model_path)
        self.transformer.resize_token_embeddings(len(tokenizer))

        self.comp_id = tokenizer.vocab["[comp_token]"]
        self.prot_id = tokenizer.vocab["[protein_token]"]

    def forward(self, input_ids, attention_mask, complex_emb, protein_emb, labels=None):
        embeds = self.transformer.transformer.word_embedding(input_ids)
        for i in range(input_ids.size(0)):
            c_idx = (input_ids[i] == self.comp_id).nonzero(as_tuple=True)[0]
            p_idx = (input_ids[i] == self.prot_id).nonzero(as_tuple=True)[0]
            if c_idx.numel():
                n = min(c_idx.numel(), complex_emb.size(1))
                embeds[i, c_idx[:n]] = complex_emb[i, :n].to(embeds.device)
            if p_idx.numel():
                n = min(p_idx.numel(), protein_emb.size(1))
                embeds[i, p_idx[:n]] = protein_emb[i, :n].to(embeds.device)
        return self.transformer(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
            labels=labels,
        )


def decode_property(
    logits: torch.Tensor, input_ids: torch.Tensor, tokenizer: ExpressionBertTokenizer
) -> torch.Tensor:
    """Decode predicted pIC50 values from model logits.

    The sequence begins with ``<pic50>`` followed by digit tokens until the
    first ``|`` token. This function extracts the predicted tokens at those
    positions and converts them into a float using
    ``tokenizer.floating_tokens_to_float``.

    Args:
        logits: Tensor of shape ``(batch, seq_len, vocab_size)``.
        input_ids: Tensor of token ids used as input.
        tokenizer: Tokenizer providing ``floating_tokens_to_float``.

    Returns:
        Tensor of shape ``(batch,)`` with the predicted floating point values.
    """
    preds = []
    pic50_id = tokenizer.vocab.get("<pic50>")
    sep_id = tokenizer.vocab.get("|")

    token_preds = torch.argmax(logits, dim=-1)

    for tok_pred, ids in zip(token_preds, input_ids):
        start_idx = (ids == pic50_id).nonzero(as_tuple=True)
        if start_idx[0].numel() == 0:
            preds.append(torch.tensor(float("nan"), device=logits.device))
            continue
        start = start_idx[0].item() + 1
        end_rel = (ids[start:] == sep_id).nonzero(as_tuple=True)
        end = start + end_rel[0].item() if end_rel[0].numel() else ids.size(0)
        pred_tokens = tokenizer.convert_ids_to_tokens(tok_pred[start:end].tolist())
        preds.append(
            torch.tensor(
                tokenizer.floating_tokens_to_float(pred_tokens), device=logits.device
            )
        )

    return torch.stack(preds)


def load_dataset(path: str) -> DockingDataset:
    records = torch.load(path)
    return DockingDataset(records)


def train(args: argparse.Namespace) -> None:
    """Fine tune the Regression Transformer."""

    vocab_path = args.tokenizer
    if os.path.isdir(vocab_path):
        vocab_path = os.path.join(vocab_path, "vocab.txt")
    tokenizer = ExpressionBertTokenizer(vocab_path)
    dataset = load_dataset(args.dataset)

    collator = Collator(tokenizer)
    val_size = max(1, int(args.val_split * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RegressionTransformer(args.model, tokenizer).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps
    )
    loss_fn = nn.MSELoss()

    best_rmse = math.inf
    os.makedirs(args.output, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        for enc, cemb, pemb, labels in train_loader:
            enc = {k: v.to(device) for k, v in enc.items()}
            cemb = cemb.to(device)
            pemb = pemb.to(device)
            labels = labels.to(device)
            output = model(**enc, complex_emb=cemb, protein_emb=pemb)
            preds = decode_property(output.logits, enc["input_ids"], tokenizer)
            loss = loss_fn(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        model.eval()
        preds, labs = [], []
        with torch.no_grad():
            for enc, cemb, pemb, labels in val_loader:
                enc = {k: v.to(device) for k, v in enc.items()}
                cemb = cemb.to(device)
                pemb = pemb.to(device)
                output = model(**enc, complex_emb=cemb, protein_emb=pemb)
                pred_vals = decode_property(output.logits, enc["input_ids"], tokenizer)
                preds.extend(pred_vals.cpu().tolist())
                labs.extend(labels.tolist())
        rmse = math.sqrt(sum((p - l) ** 2 for p, l in zip(preds, labs)) / len(labs))
        if rmse < best_rmse:
            best_rmse = rmse
            torch.save(model.state_dict(), os.path.join(args.output, "best_model.pt"))
        print(f"Epoch {epoch+1}/{args.epochs} RMSE: {rmse:.4f} (best {best_rmse:.4f})")


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune Regression Transformer on DiffDock dataset"
    )
    parser.add_argument(
        "dataset",
        help="Path to dataset file created with create_regression_transformer_input.py",
    )
    parser.add_argument("output", help="Directory to store checkpoints")
    parser.add_argument(
        "--model",
        default="models/Diffdock_RT",
        help="Path to pretrained Regression Transformer model",
    )
    parser.add_argument(
        "--tokenizer",
        default="models/Diffdock_RT",
        help="Directory containing the tokenizer vocabulary",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--val_split", type=float, default=0.2)
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    train(args)
