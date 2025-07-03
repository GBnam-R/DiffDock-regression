import os
import argparse
import numpy as np
import pandas as pd
import torch


def load_embedding(path: str) -> np.ndarray:
    """Load numpy array if file exists else return None."""
    if os.path.exists(path):
        return np.load(path)
    return None


def flatten_embedding(emb: np.ndarray) -> np.ndarray:
    """Flatten 2D embedding to 1D."""
    if emb is None:
        return None
    return emb.reshape(-1)


def prepare_dataset(data_root: str, metadata_csv: str, output_file: str) -> None:
    meta_df = pd.read_csv(metadata_csv)
    records = []
    for _, row in meta_df.iterrows():
        name = row.get("Complex_name") or row.get("complex_name")
        if not name:
            continue
        comp_dir = os.path.join(data_root, name)
        ligand_emb = load_embedding(os.path.join(comp_dir, "ligand_embedding.npy"))
        receptor_emb = load_embedding(os.path.join(comp_dir, "receptor_embedding.npy"))
        if ligand_emb is None:
            continue
        ligand_emb = flatten_embedding(ligand_emb)
        receptor_emb = flatten_embedding(receptor_emb)
        pIC50 = row.get("pIC50") or row.get("pchembl_value")
        records.append({
            "name": name,
            "compound": row.get("SMILES"),
            "protein_emb": receptor_emb,
            "complex_emb": ligand_emb,
            "pIC50": pIC50,
        })
    torch.save(records, output_file)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create dataset for regression transformer from DiffDock outputs")
    parser.add_argument("data_root", help="Path to Diffdock_data directory")
    parser.add_argument("metadata_csv", help="CSV with complex names and pIC50 values")
    parser.add_argument("output", help="Output dataset file (pt)")
    args = parser.parse_args()
    prepare_dataset(args.data_root, args.metadata_csv, args.output)


if __name__ == "__main__":
    main()
