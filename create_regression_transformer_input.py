"""Create Regression Transformer dataset from DiffDock outputs.

The resulting dataset stores tokenized sequences together with DiffDock
and ESMFold embeddings. Each entry follows the schema::

    <pic50>{digits}|<complex>[comp_token]x175|<protein>[protein_token]x175|SELFIES

where numerical values are split into digit tokens. Complex and protein
sections contain 175 placeholder tokens each. During model loading these
placeholders are replaced with the embeddings stored in the same record.
"""

import argparse
import os
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import torch
from selfies import encoder as smiles_to_selfies

from terminator.tokenization import (
    ExpressionBertTokenizer,
    PropertyTokenizer,
    SelfiesTokenizer,
)


def load_npy(path: str) -> Optional[np.ndarray]:
    """Load numpy array from ``path`` if it exists."""
    return np.load(path) if os.path.exists(path) else None


def pad_tokens(arr: np.ndarray, total: int) -> np.ndarray:
    if arr is None:
        return np.zeros((total, 256), dtype=np.float32)
    if arr.shape[0] >= total:
        return arr[:total]
    pad = np.zeros((total - arr.shape[0], arr.shape[1]), dtype=arr.dtype)
    return np.concatenate([arr, pad], axis=0)


def pool_last_dim(arr: np.ndarray, k: int) -> np.ndarray:
    new_d = arr.shape[-1] // k
    arr = arr[..., : new_d * k]
    arr = arr.reshape(*arr.shape[:-1], new_d, k)
    return arr.mean(axis=-1)


def pool_segments(arr: np.ndarray, seg: int) -> np.ndarray:
    pooled: List[np.ndarray] = []
    for i in range(0, arr.shape[0], seg):
        pooled.append(arr[i : i + seg].mean(axis=0))
    return np.stack(pooled, axis=0)


def compress_complex(emb: np.ndarray) -> np.ndarray:
    if emb is None:
        return np.zeros((0, 256), dtype=np.float32)
    out = []
    zeros = np.zeros(20, dtype=emb.dtype)
    for i in range(0, emb.shape[0], 2):
        first = emb[i]
        if i + 1 < emb.shape[0]:
            second = emb[i + 1]
        else:
            second = np.zeros_like(first)
        out.append(np.concatenate([first, zeros, second]))
    return np.stack(out, axis=0)


def prepare_complex(root: str, name: str) -> np.ndarray:
    emb = load_npy(os.path.join(root, name, "complex_embedding.npy"))
    if emb is not None and emb.ndim == 2 and emb.shape[1] == 118:
        emb = compress_complex(emb)
    else:
        emb = np.zeros((0, 256), dtype=np.float32)
    return pad_tokens(emb, 175)


def prepare_protein(root: str, name: str) -> np.ndarray:
    cls = load_npy(os.path.join(root, name, "receptor_cls_embedding.npy"))
    rec = load_npy(os.path.join(root, name, "receptor_embedding.npy"))
    tokens: List[np.ndarray] = []
    if cls is not None:
        cls = cls.reshape(-1)
        cls_avg = cls.reshape(5, 256).mean(axis=0, keepdims=True)
        tokens.append(cls_avg.squeeze(0))
        tokens.extend(cls.reshape(5, 256))
    if rec is not None:
        rec = pool_segments(rec, 20)
        rec = pool_last_dim(rec, 5)
        tokens.extend(rec)
    if tokens:
        arr = np.stack(tokens, axis=0)
    else:
        arr = np.zeros((0, 256), dtype=np.float32)
    return pad_tokens(arr, 175)


ptokenizer = PropertyTokenizer()
selfies_tokenizer = SelfiesTokenizer()


def pic50_tokens(value: float) -> List[str]:
    """Tokenize a float pIC50 value into digit tokens."""
    return ptokenizer.tokenize(f"<pic50>{value}")


def build_sequence(pIC50: float, smiles: str) -> List[str]:
    tokens = pic50_tokens(pIC50)
    tokens.append("|")
    tokens.append("<complex>")
    tokens.extend(["[comp_token]"] * 175)
    tokens.append("|")
    tokens.append("<protein>")
    tokens.extend(["[protein_token]"] * 175)
    tokens.append("|")
    tokens.extend(selfies_tokenizer.tokenize(smiles_to_selfies(smiles)))
    return tokens


def process_record(
    row: Dict[str, any], root_dir: str, tokenizer: ExpressionBertTokenizer
) -> Dict:
    pIC50 = float(row["pchembl_value_Mean"])
    smiles = row["SMILES"]
    name = f"{row['InChIKey']}_{row['Uniprot_ID']}_{row['PDB_code']}"

    complex_emb = prepare_complex(root_dir, name)
    protein_emb = prepare_protein(root_dir, name)

    seq_tokens = build_sequence(pIC50, smiles)
    input_ids = tokenizer.convert_tokens_to_ids(seq_tokens)

    return {
        "name": name,
        "input_ids": input_ids,
        "complex_emb": complex_emb,
        "protein_emb": protein_emb,
        "pIC50": pIC50,
    }


def get_parser() -> argparse.ArgumentParser:
    """Return CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Create Regression Transformer input from DiffDock results"
    )
    parser.add_argument(
        "--protein_ligand_csv",
        type=str,
        required=True,
        help="CSV file with docking results and metadata",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Directory with DiffDock outputs where dataset will be saved",
    )
    parser.add_argument(
        "--dataset_file",
        type=str,
        default="dataset.pt",
        help="Name of the generated dataset file",
    )
    return parser


def main(args: argparse.Namespace) -> None:
    diffdock_dir = args.out_dir
    csv_file = args.protein_ligand_csv
    output_path = (
        args.dataset_file
        if os.path.isabs(args.dataset_file)
        else os.path.join(diffdock_dir, args.dataset_file)
    )

    tokenizer = ExpressionBertTokenizer.from_pretrained("model/Diffdock_RT")
    added = tokenizer.add_tokens(
        ["[comp_token]", "[protein_token]", "<complex>", "<protein>", "<pic50>"]
    )
    if added:
        print(f"Added {added} tokens to tokenizer")

    df = pd.read_csv(csv_file)
    records = [process_record(row, diffdock_dir, tokenizer) for _, row in df.iterrows()]
    torch.save(records, output_path)


if __name__ == "__main__":
    parser = get_parser()
    main(parser.parse_args())
