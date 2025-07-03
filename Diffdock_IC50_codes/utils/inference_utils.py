import copy
import os
import pickle
import numpy as np

import torch
from Bio.PDB import PDBParser
from esm import FastaBatchedDataset, pretrained
from rdkit.Chem import AddHs, MolFromSmiles
from torch_geometric.data import Dataset, HeteroData
import esm

from datasets.constants import three_to_one
from datasets.process_mols import generate_conformer, read_molecule, get_lig_graph_with_matching, moad_extract_receptor_structure


def get_sequences_from_pdbfile(file_path):
    biopython_parser = PDBParser()
    structure = biopython_parser.get_structure('random_id', file_path)
    structure = structure[0]
    sequence = None
    for i, chain in enumerate(structure):
        seq = ''
        for res_idx, residue in enumerate(chain):
            if residue.get_resname() == 'HOH':
                continue
            residue_coords = []
            c_alpha, n, c = None, None, None
            for atom in residue:
                if atom.name == 'CA':
                    c_alpha = list(atom.get_vector())
                if atom.name == 'N':
                    n = list(atom.get_vector())
                if atom.name == 'C':
                    c = list(atom.get_vector())
            if c_alpha != None and n != None and c != None:  # only append residue if it is an amino acid
                try:
                    seq += three_to_one[residue.get_resname()]
                except Exception as e:
                    seq += '-'
                    print("encountered unknown AA: ", residue.get_resname(), ' in the complex. Replacing it with a dash - .')

        if sequence is None:
            sequence = seq
        else:
            sequence += (":" + seq)

    return sequence


def set_nones(l):
    return [s if str(s) != 'nan' else None for s in l]


def get_sequences(protein_files, protein_sequences, prefer_pdb: bool = False):
    """Return the sequence for each complex.

    Parameters
    ----------
    protein_files : list[str]
        Paths to protein pdb files.
    protein_sequences : list[str]
        Optional sequences provided by the user.
    prefer_pdb : bool, default False
        If True, use the sequence extracted from the pdb file whenever
        a pdb path is available. Otherwise prefer the provided sequence.
    """

    new_sequences = []
    for pdb_file, seq in zip(protein_files, protein_sequences):
        if prefer_pdb and pdb_file is not None:
            new_sequences.append(get_sequences_from_pdbfile(pdb_file))
        elif seq is not None:
            new_sequences.append(seq)
        elif pdb_file is not None:
            new_sequences.append(get_sequences_from_pdbfile(pdb_file))
        else:
            new_sequences.append(None)

    return new_sequences


def compute_ESM_embeddings(model, alphabet, labels, sequences):
    """Compute per-residue ESMFold embeddings and CLS token representations."""
    # settings used
    toks_per_batch = 4096
    repr_layers = [33]
    truncation_seq_length = 10000

    dataset = FastaBatchedDataset(labels, sequences)
    batches = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=alphabet.get_batch_converter(truncation_seq_length), batch_sampler=batches
    )

    assert all(-(model.num_layers + 1) <= i <= model.num_layers for i in repr_layers)
    repr_layers = [(i + model.num_layers + 1) % (model.num_layers + 1) for i in repr_layers]
    embeddings = {}
    cls_tokens = {}

    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            print(f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)")
            if torch.cuda.is_available():
                toks = toks.to(device="cuda", non_blocking=True)

            out = model(toks, repr_layers=repr_layers, return_contacts=False)
            representations = {layer: t.to(device="cpu") for layer, t in out["representations"].items()}

            for i, label in enumerate(labels):
                truncate_len = min(truncation_seq_length, len(strs[i]))
                rep = representations[33][i]
                cls_tokens[label] = rep[0].clone()
                embeddings[label] = rep[1: truncate_len + 1].clone()
    return embeddings, cls_tokens


def generate_ESM_structure(model, filename, sequence):
    model.set_chunk_size(256)
    chunk_size = 256
    output = None

    while output is None:
        try:
            with torch.no_grad():
                output = model.infer_pdb(sequence)

            with open(filename, "w") as f:
                f.write(output)
                print("saved", filename)
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory on chunk_size', chunk_size)
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                chunk_size = chunk_size // 2
                if chunk_size > 2:
                    model.set_chunk_size(chunk_size)
                else:
                    print("Not enough memory for ESMFold")
                    break
            else:
                raise e
    return output is not None


class InferenceDataset(Dataset):
    def __init__(self, out_dir, complex_names, protein_files, ligand_descriptions, protein_sequences, lm_embeddings,
                 receptor_radius=30, c_alpha_max_neighbors=None, precomputed_lm_embeddings=None,
                 remove_hs=False, all_atoms=False, atom_radius=5, atom_max_neighbors=None, knn_only_graph=False):

        super(InferenceDataset, self).__init__()
        self.receptor_radius = receptor_radius
        self.c_alpha_max_neighbors = c_alpha_max_neighbors
        self.remove_hs = remove_hs
        self.all_atoms = all_atoms
        self.atom_radius, self.atom_max_neighbors = atom_radius, atom_max_neighbors
        self.knn_only_graph = knn_only_graph

        self.complex_names = complex_names
        self.protein_files = protein_files
        self.ligand_descriptions = ligand_descriptions
        self.protein_sequences = protein_sequences

        # generate LM embeddings
        if lm_embeddings and (precomputed_lm_embeddings is None or precomputed_lm_embeddings[0] is None):
            print("Generating ESM language model embeddings")
            model_location = "esm2_t33_650M_UR50D"
            model, alphabet = pretrained.load_model_and_alphabet(model_location)
            model.eval()
            if torch.cuda.is_available():
                model = model.cuda()

            docking_sequences = get_sequences(protein_files, protein_sequences, prefer_pdb=True)
            save_sequences = get_sequences(protein_files, protein_sequences, prefer_pdb=False)

            labels, sequences = [], []
            for i in range(len(docking_sequences)):
                s = docking_sequences[i].split(':')
                sequences.extend(s)
                labels.extend([f'{complex_names[i]}chain{j}' for j in range(len(s))])
            docking_embeds, _ = compute_ESM_embeddings(model, alphabet, labels, sequences)

            labels, sequences = [], []
            for i in range(len(save_sequences)):
                s = save_sequences[i].split(':')
                sequences.extend(s)
                labels.extend([f'{complex_names[i]}chain{j}' for j in range(len(s))])
            save_embeds, cls_tokens = compute_ESM_embeddings(model, alphabet, labels, sequences)

            self.model_embeddings = []
            self.save_embeddings = []
            self.cls_embeddings = []
            for i in range(len(docking_sequences)):
                d_chains = docking_sequences[i].split(':')
                s_chains = save_sequences[i].split(':')
                self.model_embeddings.append([docking_embeds[f'{complex_names[i]}chain{j}'] for j in range(len(d_chains))])
                self.save_embeddings.append([save_embeds[f'{complex_names[i]}chain{j}'] for j in range(len(s_chains))])
                self.cls_embeddings.append([cls_tokens[f'{complex_names[i]}chain{j}'] for j in range(len(s_chains))])

        elif not lm_embeddings:
            self.model_embeddings = [None] * len(self.complex_names)
            self.save_embeddings = [None] * len(self.complex_names)
            self.cls_embeddings = [None] * len(self.complex_names)

        else:
            self.model_embeddings = precomputed_lm_embeddings
            self.save_embeddings = precomputed_lm_embeddings
            self.cls_embeddings = [None] * len(self.complex_names)

        # generate structures with ESMFold
        if None in protein_files:
            print("generating missing structures with ESMFold")
            model = esm.pretrained.esmfold_v1()
            model = model.eval().cuda()

            for i in range(len(protein_files)):
                if protein_files[i] is None:
                    self.protein_files[i] = f"{out_dir}/{complex_names[i]}/{complex_names[i]}_esmfold.pdb"
                    if not os.path.exists(self.protein_files[i]):
                        print("generating", self.protein_files[i])
                        generate_ESM_structure(model, self.protein_files[i], protein_sequences[i])

    def len(self):
        return len(self.complex_names)

    def get(self, idx):

        name = self.complex_names[idx]
        protein_file = self.protein_files[idx]
        ligand_description = self.ligand_descriptions[idx]
        model_embedding = self.model_embeddings[idx]
        save_embedding = self.save_embeddings[idx]
        cls_embedding = self.cls_embeddings[idx] if hasattr(self, 'cls_embeddings') else None

        # build the pytorch geometric heterogeneous graph
        complex_graph = HeteroData()
        complex_graph['name'] = name

        # parse the ligand, either from file or smile
        try:
            mol = MolFromSmiles(ligand_description)  # check if it is a smiles or a path

            if mol is not None:
                mol = AddHs(mol)
                generate_conformer(mol)
            else:
                mol = read_molecule(ligand_description, remove_hs=False, sanitize=True)
                if mol is None:
                    raise Exception('RDKit could not read the molecule ', ligand_description)
                mol.RemoveAllConformers()
                mol = AddHs(mol)
                generate_conformer(mol)
        except Exception as e:
            print('Failed to read molecule ', ligand_description, ' We are skipping it. The reason is the exception: ', e)
            complex_graph['success'] = False
            return complex_graph

        try:
            # parse the receptor from the pdb file
            get_lig_graph_with_matching(mol, complex_graph, popsize=None, maxiter=None, matching=False, keep_original=False,
                                        num_conformers=1, remove_hs=self.remove_hs)

            moad_extract_receptor_structure(
                path=os.path.join(protein_file),
                complex_graph=complex_graph,
                neighbor_cutoff=self.receptor_radius,
                max_neighbors=self.c_alpha_max_neighbors,
                lm_embeddings=model_embedding,
                knn_only_graph=self.knn_only_graph,
                all_atoms=self.all_atoms,
                atom_cutoff=self.atom_radius,
                atom_max_neighbors=self.atom_max_neighbors)

            if save_embedding is not None:
                complex_graph['receptor'].input_lm_embeddings = torch.tensor(np.concatenate(save_embedding, axis=0))

            if cls_embedding is not None:
                complex_graph['receptor'].cls_embedding = torch.stack(
                    [torch.tensor(e) for e in cls_embedding]
                )

        except Exception as e:
            print(f'Skipping {name} because of the error:')
            print(e)
            complex_graph['success'] = False
            return complex_graph

        protein_center = torch.mean(complex_graph['receptor'].pos, dim=0, keepdim=True)
        complex_graph['receptor'].pos -= protein_center
        if self.all_atoms:
            complex_graph['atom'].pos -= protein_center

        ligand_center = torch.mean(complex_graph['ligand'].pos, dim=0, keepdim=True)
        complex_graph['ligand'].pos -= ligand_center

        complex_graph.original_center = protein_center
        complex_graph.mol = mol
        complex_graph['success'] = True
        return complex_graph
