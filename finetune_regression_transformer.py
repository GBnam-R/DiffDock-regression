import os
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import torch
from torch import nn
from torch.utils.data import Dataset, random_split
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    XLNetLMHeadModel,
)

from terminator.collators import TRAIN_COLLATORS
from terminator.tokenization import ExpressionBertTokenizer
from terminator.trainer import CustomTrainer, get_trainer_dict


class DockingDataset(Dataset):
    """Dataset providing token ids and embeddings."""

    def __init__(self, records: List[Dict[str, Any]]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        return {
            "input_ids": rec["input_ids"],
            "complex_emb": rec["complex_emb"],
            "protein_emb": rec["protein_emb"],
            "prop_value": float(rec["pIC50"]),
        }


class EmbeddingCollator:
    """Wrapper collator injecting DiffDock embeddings."""

    def __init__(self, base_collator):
        self.base_collator = base_collator
        self.num_primed = getattr(base_collator, "num_primed", 1)
        self.property_tokens = getattr(base_collator, "property_tokens", None)

    def __call__(self, batch: List[Dict[str, Any]]):
        inputs = [b["input_ids"] for b in batch]
        collated = self.base_collator(inputs)

        cemb = torch.stack([torch.tensor(b["complex_emb"], dtype=torch.float32) for b in batch])
        pemb = torch.stack([torch.tensor(b["protein_emb"], dtype=torch.float32) for b in batch])

        cemb = cemb.repeat_interleave(self.num_primed, dim=0)
        pemb = pemb.repeat_interleave(self.num_primed, dim=0)

        collated["complex_emb"] = cemb
        collated["protein_emb"] = pemb
        collated["prop_value"] = torch.tensor([b["prop_value"] for b in batch], dtype=torch.float32).repeat_interleave(
            self.num_primed
        )

        return collated


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


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="models/Diffdock_RT")
    tokenizer_name: Optional[str] = field(default=None)


@dataclass
class DataArguments:
    dataset_path: str = field(metadata={"help": "Path to prepared dataset"})
    val_split: float = field(default=0.2)
    training_config_path: Optional[str] = field(default=None)
    plm_probability: float = field(default=1 / 6)
    max_span_length: int = field(default=5)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.local_rank != -1 and not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend)
        torch.cuda.set_device(training_args.local_rank)

    tokenizer_path = model_args.tokenizer_name or model_args.model_name_or_path
    tokenizer = ExpressionBertTokenizer.from_pretrained(tokenizer_path)

    dataset = load_dataset(data_args.dataset_path)
    val_size = max(1, int(data_args.val_split * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_config: Dict[str, Any] = {}
    if data_args.training_config_path:
        with open(data_args.training_config_path) as f:
            train_config = json.load(f)

    if train_config.get("alternate_tasks", False):
        base_pp = TRAIN_COLLATORS["property"](
            tokenizer=tokenizer,
            property_tokens=train_config["property_tokens"],
            num_tokens_to_mask=train_config.get("num_tokens_to_mask", None),
            mask_token_order=train_config.get("mask_token_order", None),
        )
        base_cg = TRAIN_COLLATORS[train_config["cg_collator"]](
            tokenizer=tokenizer, **train_config["cg_collator_params"]
        )
        data_collator = EmbeddingCollator(base_pp)
        alternating_collator = EmbeddingCollator(base_cg)
    else:
        task = train_config.get("task", "proponly")
        if task == "gen_only":
            base = TRAIN_COLLATORS[train_config["cg_collator"]](
                tokenizer=tokenizer, **train_config["cg_collator_params"]
            )
        elif task == "plm":
            from transformers import DataCollatorForPermutationLanguageModeling

            base = DataCollatorForPermutationLanguageModeling(
                tokenizer=tokenizer,
                plm_probability=data_args.plm_probability,
                max_span_length=data_args.max_span_length,
            )
        else:  # proponly
            base = TRAIN_COLLATORS["property"](
                tokenizer=tokenizer,
                property_tokens=train_config.get("property_tokens", ["<pic50>"]),
                num_tokens_to_mask=train_config.get("num_tokens_to_mask", None),
                mask_token_order=train_config.get("mask_token_order", None),
            )
        data_collator = EmbeddingCollator(base)
        alternating_collator = None

    model = RegressionTransformer(model_args.model_name_or_path, tokenizer)

    custom_trainer_params = get_trainer_dict({})
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        alternating_collator=alternating_collator,
        train_config=train_config,
        **custom_trainer_params,
    )

    trainer.train()

    if training_args.output_dir is not None:
        trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()
