import os
import copy
import yaml
import numpy as np
import torch
import pandas as pd
from argparse import ArgumentParser, Namespace, FileType
from functools import partial
from torch_geometric.loader import DataLoader

from datasets.process_mols import write_mol_with_coords
from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl, get_t_schedule
from utils.inference_utils import InferenceDataset, set_nones
from utils.sampling import randomize_position, sampling
from utils.utils import get_model
from rdkit.Chem import RemoveAllHs


def get_parser():
    parser = ArgumentParser()
    parser.add_argument('--config', type=FileType('r'), default='default_inference_args.yaml')
    parser.add_argument('--protein_ligand_csv', type=str, default=None,
                        help='CSV with columns complex_name, ligand_description, protein_path, protein_sequence')
    parser.add_argument('--protein_path', type=str, default=None)
    parser.add_argument('--ligand_description', type=str, default=None)
    parser.add_argument('--protein_sequence', type=str, default=None)
    parser.add_argument('--complex_name', type=str, default='complex')
    parser.add_argument('--out_dir', type=str, default='results')
    parser.add_argument('--confidence_model_dir', type=str, default='./workdir/v1.1/confidence_model')
    parser.add_argument('--confidence_ckpt', type=str, default='best_model_epoch75.pt')
    parser.add_argument('--old_confidence_model', action='store_true', default=True)
    return parser


def main(args: Namespace) -> None:
    if args.config:
        config_dict = yaml.load(args.config, Loader=yaml.FullLoader)
        arg_dict = args.__dict__
        for key, value in config_dict.items():
            arg_dict.setdefault(key, value)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(f'{args.model_dir}/model_parameters.yml') as f:
        score_model_args = Namespace(**yaml.full_load(f))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t_to_sigma = partial(t_to_sigma_compl, args=score_model_args)

    model = get_model(score_model_args, device, t_to_sigma=t_to_sigma, no_parallel=True, old=args.old_score_model)
    state_dict = torch.load(f'{args.model_dir}/{args.ckpt}', map_location='cpu')
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()

    if args.protein_ligand_csv is not None:
        df = pd.read_csv(args.protein_ligand_csv)
        complex_names = set_nones(df["complex_name"].tolist())
        protein_files = set_nones(df["protein_path"].tolist())
        ligand_descs = set_nones(df["ligand_description"].tolist())
        protein_seqs = set_nones(df.get("protein_sequence", []).tolist()) if "protein_sequence" in df.columns else [None] * len(df)
    else:
        complex_names = [args.complex_name]
        protein_files = [args.protein_path]
        ligand_descs = [args.ligand_description]
        protein_seqs = [args.protein_sequence]

    complex_names = [n if n is not None else f"complex_{i}" for i, n in enumerate(complex_names)]

    dataset = InferenceDataset(
        out_dir=args.out_dir,
        complex_names=complex_names,
        protein_files=protein_files,
        ligand_descriptions=ligand_descs,
        protein_sequences=protein_seqs,
        lm_embeddings=True,
        receptor_radius=score_model_args.receptor_radius,
        remove_hs=score_model_args.remove_hs,
        c_alpha_max_neighbors=score_model_args.c_alpha_max_neighbors,
        all_atoms=score_model_args.all_atoms,
        atom_radius=score_model_args.atom_radius,
        atom_max_neighbors=score_model_args.atom_max_neighbors,
        knn_only_graph=not getattr(score_model_args, 'not_knn_only_graph', False),
    )

    if args.confidence_model_dir is not None:
        with open(f'{args.confidence_model_dir}/model_parameters.yml') as f:
            confidence_args = Namespace(**yaml.full_load(f))
        confidence_dataset = InferenceDataset(
            out_dir=args.out_dir,
            complex_names=complex_names,
            protein_files=protein_files,
            ligand_descriptions=ligand_descs,
            protein_sequences=protein_seqs,
            lm_embeddings=True,
            receptor_radius=confidence_args.receptor_radius,
            remove_hs=confidence_args.remove_hs,
            c_alpha_max_neighbors=confidence_args.c_alpha_max_neighbors,
            all_atoms=confidence_args.all_atoms,
            atom_radius=confidence_args.atom_radius,
            atom_max_neighbors=confidence_args.atom_max_neighbors,
            precomputed_lm_embeddings=dataset.model_embeddings,
            knn_only_graph=not getattr(confidence_args, 'not_knn_only_graph', False),
        )
        confidence_model = get_model(
            confidence_args,
            device,
            t_to_sigma=t_to_sigma,
            no_parallel=True,
            confidence_mode=True,
            old=args.old_confidence_model,
        )
        state_dict = torch.load(f'{args.confidence_model_dir}/{args.confidence_ckpt}', map_location='cpu')
        confidence_model.load_state_dict(state_dict, strict=True)
        confidence_model = confidence_model.to(device)
        confidence_model.eval()
    else:
        confidence_dataset = None
        confidence_model = None
        confidence_args = None

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    confidence_loader = iter(DataLoader(confidence_dataset, batch_size=1, shuffle=False)) if confidence_dataset is not None else None

    tr_schedule = get_t_schedule(sigma_schedule='expbeta', inference_steps=args.inference_steps)

    for i, orig_complex_graph in enumerate(loader):
        if not orig_complex_graph.success[0]:
            continue
        data_list = [copy.deepcopy(orig_complex_graph) for _ in range(args.samples_per_complex)]
        randomize_position(data_list, score_model_args.no_torsion, False, score_model_args.tr_sigma_max)

        if confidence_loader is not None:
            confidence_complex_graph = next(confidence_loader)
            if not confidence_complex_graph.success:
                continue
            confidence_data_list = [copy.deepcopy(confidence_complex_graph) for _ in range(args.samples_per_complex)]
        else:
            confidence_data_list = None

        name_field = orig_complex_graph['name']
        complex_name = name_field if isinstance(name_field, str) else name_field[0]

        try:
            data_list, confidence, final_embedding, final_complex_graph = sampling(
                data_list=data_list,
                model=model,
                inference_steps=args.inference_steps,
                tr_schedule=tr_schedule,
                rot_schedule=tr_schedule,
                tor_schedule=tr_schedule,
                device=device,
                t_to_sigma=t_to_sigma,
                model_args=score_model_args,
                confidence_model=confidence_model,
                confidence_data_list=confidence_data_list,
                confidence_model_args=confidence_args,
            )
        except RuntimeError as e:
            if 'svd' in str(e).lower():
                print(f"SVD failed for {complex_name}: {e}. Skipping.")
            else:
                print(f"Runtime error for {complex_name}: {e}. Skipping.")
            continue

        ligand_pos = [
            complex_graph['ligand'].pos.cpu().numpy() + orig_complex_graph.original_center.cpu().numpy()
            for complex_graph in data_list
        ]

        if confidence is not None:
            if confidence.dim() > 1:
                confidence = confidence[:, 0]
            confidence = confidence.cpu().numpy()
            order = np.argsort(confidence)[::-1]
            confidence = confidence[order]
            ligand_pos = [ligand_pos[i] for i in order]
        else:
            confidence = [float('nan')] * len(ligand_pos)

        write_dir = os.path.join(args.out_dir, complex_name)
        os.makedirs(write_dir, exist_ok=True)
        mol_pred = copy.deepcopy(orig_complex_graph.mol[0])
        if score_model_args.remove_hs:
            mol_pred = RemoveAllHs(mol_pred)
        for rank, pos in enumerate(ligand_pos):
            write_mol_with_coords(
                mol_pred,
                pos,
                os.path.join(write_dir, f'rank{rank+1}_confidence_{confidence[rank]:.2f}.sdf'),
            )

        np.save(os.path.join(write_dir, 'complex_embedding.npy'), final_embedding.cpu().numpy())
        if hasattr(orig_complex_graph['receptor'], 'input_lm_embeddings'):
            rec_emb = orig_complex_graph['receptor'].input_lm_embeddings
        elif hasattr(orig_complex_graph['receptor'], 'lm_embeddings'):
            rec_emb = orig_complex_graph['receptor'].lm_embeddings
        else:
            rec_emb = None

        if rec_emb is not None:
            np.save(os.path.join(write_dir, 'receptor_embedding.npy'), rec_emb.cpu().numpy())
            if hasattr(orig_complex_graph['receptor'], 'cls_embedding'):
                cls_emb = orig_complex_graph['receptor'].cls_embedding
                if cls_emb.dim() > 1:
                    cls_emb = cls_emb.mean(dim=0, keepdim=True)
                np.save(os.path.join(write_dir, 'receptor_cls_embedding.npy'), cls_emb.cpu().numpy())


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
