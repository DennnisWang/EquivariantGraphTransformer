import logging
import os.path

import dgl
import networkx as nx
import numpy as np
import re
import selfies as sf
import sys
import time
import torch
from dgllife.data.uspto import atom_types
from dgllife.utils import get_mol_3d_coordinates, mol_to_bigraph, BaseAtomFeaturizer
from rdkit import Chem
from rdkit.Chem import AllChem
from torch.utils.data import Dataset
from typing import Dict, List, Tuple

from dgllife.utils.featurizers import *
from functools import partial


from utils.chem_utils import ATOM_FDIM, BOND_FDIM, get_atom_features_sparse, get_bond_features
from utils.rxn_graphs import RxnGraph

# hv: 125
node_featurizer_egnn = BaseAtomFeaturizer({
    'hv': ConcatFeaturizer(
        [partial(atom_type_one_hot,
                 allowable_set=atom_types, encode_unknown=True),
         partial(atom_degree_one_hot, encode_unknown=True),
         partial(atom_formal_charge_one_hot, encode_unknown=True),
         partial(atom_explicit_valence_one_hot, encode_unknown=True),
         partial(atom_hybridization_one_hot, encode_unknown=True),
         partial(atom_total_num_H_one_hot, encode_unknown=True),
         partial(atom_chirality_type_one_hot, encode_unknown=False),
         partial(atom_num_radical_electrons_one_hot, encode_unknown=True),
         partial(atom_chiral_tag_one_hot, encode_unknown=True),
         partial(atom_implicit_valence_one_hot, encode_unknown=True),
         partial(atom_is_aromatic_one_hot, encode_unknown=True),
         atom_is_aromatic]
    )
})
#he 17
edge_featurizer_egnn = BaseBondFeaturizer({
    'he': ConcatFeaturizer([
        bond_type_one_hot, bond_is_conjugated_one_hot, bond_is_in_ring_one_hot,bond_stereo_one_hot,bond_direction_one_hot]
    )
})

def tokenize_selfies_from_smiles(smi: str) -> str:
    encoded_selfies = sf.encoder(smi)
    tokens = list(sf.split_selfies(encoded_selfies))
    assert encoded_selfies == "".join(tokens)

    return " ".join(tokens)


def tokenize_smiles(smi: str) -> str:
    pattern = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == "".join(tokens), f"Tokenization mismatch. smi: {smi}, tokens: {tokens}"

    return " ".join(tokens)


def canonicalize_smiles(smiles, remove_atom_number=False, trim=True, suppress_warning=False):
    cano_smiles = ""

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        cano_smiles = ""

    else:
        if trim and mol.GetNumHeavyAtoms() < 2:
            if not suppress_warning:
                logging.info(f"Problematic smiles: {smiles}, setting it to 'CC'")
            cano_smiles = "CC"  # TODO: hardcode to ignore
        else:
            if remove_atom_number:
                [a.ClearProp('molAtomMapNumber') for a in mol.GetAtoms()]
            cano_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)

    return cano_smiles


def len2idx(lens) -> np.ndarray:
    # end_indices = np.cumsum(np.concatenate(lens, axis=0))
    end_indices = np.cumsum(lens)
    start_indices = np.concatenate([[0], end_indices[:-1]], axis=0)
    indices = np.stack([start_indices, end_indices], axis=1)

    return indices


class S2SBatch:
    def __init__(self,
                 src_token_ids: torch.Tensor,
                 src_lengths: torch.Tensor,
                 tgt_token_ids: torch.Tensor,
                 tgt_lengths: torch.Tensor):
        self.src_token_ids = src_token_ids
        self.src_lengths = src_lengths
        self.tgt_token_ids = tgt_token_ids
        self.tgt_lengths = tgt_lengths

        self.size = len(src_lengths)

    def to(self, device):
        self.src_token_ids = self.src_token_ids.to(device)
        self.src_lengths = self.src_lengths.to(device)
        self.tgt_token_ids = self.tgt_token_ids.to(device)
        self.tgt_lengths = self.tgt_lengths.to(device)

    def pin_memory(self):
        self.src_token_ids = self.src_token_ids.pin_memory()
        self.src_lengths = self.src_lengths.pin_memory()
        self.tgt_token_ids = self.tgt_token_ids.pin_memory()
        self.tgt_lengths = self.tgt_lengths.pin_memory()

        return self

    def log_tensor_shape(self):
        logging.info(f"src_token_ids: {self.src_token_ids.shape}, "
                     f"src_lengths: {self.src_lengths}, "
                     f"tgt_token_ids: {self.tgt_token_ids.shape}, "
                     f"tgt_lengths: {self.tgt_lengths}")


class S2SDataset(Dataset):
    def __init__(self, args, file: str):
        self.args = args

        self.src_token_ids = []
        self.src_lens = []
        self.tgt_token_ids = []
        self.tgt_lens = []

        self.data_indices = []
        self.batch_sizes = []
        self.batch_starts = []
        self.batch_ends = []

        logging.info(f"Loading preprocessed features from {file}")
        feat = np.load(file)
        for attr in ["src_token_ids", "src_lens", "tgt_token_ids", "tgt_lens"]:
            setattr(self, attr, feat[attr])

        assert len(self.src_token_ids) == len(self.src_lens) == len(self.tgt_token_ids) == len(self.tgt_lens), \
            f"Lengths of source and target mismatch!"

        self.data_size = len(self.src_token_ids)
        self.data_indices = np.arange(self.data_size)

        logging.info(f"Loaded and initialized S2SDataset, size: {self.data_size}")

    def sort(self):
        start = time.time()

        logging.info(f"Calling S2SDataset.sort()")
        sys.stdout.flush()
        self.data_indices = np.argsort(self.src_lens + self.tgt_lens)

        logging.info(f"Done, time: {time.time() - start: .2f} s")
        sys.stdout.flush()

    def shuffle_in_bucket(self, bucket_size: int):
        start = time.time()

        logging.info(f"Calling S2SDataset.shuffle_in_bucket()")
        sys.stdout.flush()

        for i in range(0, self.data_size, bucket_size):
            np.random.shuffle(self.data_indices[i:i + bucket_size])

        logging.info(f"Done, time: {time.time() - start: .2f} s")
        sys.stdout.flush()

    def batch(self, batch_type: str, batch_size: int):
        start = time.time()

        logging.info(f"Calling S2SDataset.batch()")
        sys.stdout.flush()

        self.batch_sizes = []

        if batch_type == "samples":
            raise NotImplementedError

        elif batch_type == "atoms":
            raise NotImplementedError

        elif batch_type == "tokens":
            sample_size = 0
            max_batch_src_len = 0
            max_batch_tgt_len = 0

            for data_idx in self.data_indices:
                src_len = self.src_lens[data_idx]
                tgt_len = self.tgt_lens[data_idx]

                max_batch_src_len = max(src_len, max_batch_src_len)
                max_batch_tgt_len = max(tgt_len, max_batch_tgt_len)
                while self.args.enable_amp and not max_batch_src_len % 8 == 0:  # for amp
                    max_batch_src_len += 1
                while self.args.enable_amp and not max_batch_tgt_len % 8 == 0:  # for amp
                    max_batch_tgt_len += 1

                if (max_batch_src_len + max_batch_tgt_len) * (sample_size + 1) <= batch_size:
                    sample_size += 1
                elif self.args.enable_amp and not sample_size % 8 == 0:
                    sample_size += 1
                else:
                    self.batch_sizes.append(sample_size)

                    sample_size = 1
                    max_batch_src_len = src_len
                    max_batch_tgt_len = tgt_len
                    while self.args.enable_amp and not max_batch_src_len % 8 == 0:  # for amp

                        max_batch_src_len += 1
                    while self.args.enable_amp and not max_batch_tgt_len % 8 == 0:  # for amp
                        max_batch_tgt_len += 1

            # lastly
            self.batch_sizes.append(sample_size)
            self.batch_sizes = np.array(self.batch_sizes)
            assert np.sum(self.batch_sizes) == self.data_size, \
                f"Size mismatch! Data size: {self.data_size}, sum batch sizes: {np.sum(self.batch_sizes)}"

            self.batch_ends = np.cumsum(self.batch_sizes)
            self.batch_starts = np.concatenate([[0], self.batch_ends[:-1]])

        else:
            raise ValueError(f"batch_type {batch_type} not supported!")

        logging.info(f"Done, time: {time.time() - start: .2f} s, total batches: {self.__len__()}")
        sys.stdout.flush()

    def __getitem__(self, index: int) -> S2SBatch:
        batch_start = self.batch_starts[index]
        batch_end = self.batch_ends[index]

        data_indices = self.data_indices[batch_start:batch_end]

        # collating, essentially
        src_token_ids = self.src_token_ids[data_indices]
        src_lengths = self.src_lens[data_indices]
        tgt_token_ids = self.tgt_token_ids[data_indices]
        tgt_lengths = self.tgt_lens[data_indices]

        src_token_ids = src_token_ids[:, :max(src_lengths)]
        tgt_token_ids = tgt_token_ids[:, :max(tgt_lengths)]

        src_token_ids = torch.as_tensor(src_token_ids, dtype=torch.long)
        tgt_token_ids = torch.as_tensor(tgt_token_ids, dtype=torch.long)
        src_lengths = torch.tensor(src_lengths, dtype=torch.long)
        tgt_lengths = torch.tensor(tgt_lengths, dtype=torch.long)

        s2s_batch = S2SBatch(
            src_token_ids=src_token_ids,
            src_lengths=src_lengths,
            tgt_token_ids=tgt_token_ids,
            tgt_lengths=tgt_lengths
        )
        # s2s_batch.log_tensor_shape()
        return s2s_batch

    def __len__(self):
        return len(self.batch_sizes)


class G2SBatch:
    def __init__(self,
                 atom_size: list,
                 batch_graph: dgl.DGLGraph,
                 tgt_token_ids: torch.Tensor,
                 tgt_lengths: torch.Tensor,
                 batch_input : dict = None,
                 distances: torch.Tensor = None
                 ):
        self.batch_graph = batch_graph
        self.tgt_token_ids = tgt_token_ids
        self.tgt_lengths = tgt_lengths
        self.distances = distances
        self.atom_size = atom_size
        self.batch_input = batch_input
        self.size = len(atom_size)

    def to(self, device):
        self.batch_graph = self.batch_graph.to(device)
        if self.tgt_token_ids is not None:
            self.tgt_token_ids = self.tgt_token_ids.to(device)
        if self.tgt_lengths is not None:
            self.tgt_lengths = self.tgt_lengths.to(device)

        #to device
        if self.batch_input is not None:
            net_input = self.batch_input
            self.batch_input = {k: v.to(device) for k, v in net_input.items()}

        if self.distances is not None:
            self.distances = self.distances.to(device)

    def pin_memory(self):
        self.batch_graph = self.batch_graph.pin_memory_()
        self.tgt_token_ids = self.tgt_token_ids.pin_memory()
        self.tgt_lengths = self.tgt_lengths.pin_memory()

        if self.distances is not None:
            self.distances = self.distances.pin_memory()

        return self

    def log_tensor_shape(self):
        logging.info(f"graph batch_size:{self.batch_graph.batch_size} "
                     f"tgt_token_ids: {self.tgt_token_ids.shape}, "
                     f"tgt_lengths: {self.tgt_lengths}")




class G2SDataset(Dataset):
    def __init__(self, args, file: str):
        self.args = args

        self.src_token_ids = []  # loaded but not batched
        self.src_lens = []
        self.tgt_token_ids = []
        self.tgt_lens = []

        self.data_indices = []
        self.batch_sizes = []
        self.batch_starts = []
        self.batch_ends = []

        self.graphs = []

        self.vocab = load_vocab(args.vocab_file)
        self.vocab_tokens = [k for k, v in sorted(self.vocab.items(), key=lambda tup: tup[1])]

        logging.info(f"Loading preprocessed features from {file}")
        feat = np.load(file)
        for attr in ["src_token_ids", "src_lens", "tgt_token_ids", "tgt_lens"]:
            setattr(self, attr, feat[attr])
        file = file.replace('.npz', '.bin')
        logging.info(f"Loading preprocessed  graph features from {file}")
        self.graphs = dgl.load_graphs(file)[0]



        assert len(self.src_token_ids) == len(self.src_lens) == \
               len(self.tgt_token_ids) == len(self.tgt_lens) == \
               len(self.graphs) == len(self.src_token_ids) , f"Lengths of source and target mismatch!"

        self.data_size = len(self.src_token_ids)
        self.data_indices = np.arange(self.data_size)

        logging.info(f"Loaded and initialized G2SDataset, size: {self.data_size}")

    def sort(self):
        if self.args.verbose:
            start = time.time()

            logging.info(f"Calling G2SDataset.sort()")
            sys.stdout.flush()
            self.data_indices = np.argsort(self.src_lens)

            logging.info(f"Done, time: {time.time() - start: .2f} s")
            sys.stdout.flush()

        else:
            self.data_indices = np.argsort(self.src_lens)

    def shuffle_in_bucket(self, bucket_size: int):
        if self.args.verbose:
            start = time.time()

            logging.info(f"Calling G2SDataset.shuffle_in_bucket()")
            sys.stdout.flush()

            for i in range(0, self.data_size, bucket_size):
                np.random.shuffle(self.data_indices[i:i + bucket_size])

            logging.info(f"Done, time: {time.time() - start: .2f} s")
            sys.stdout.flush()

        else:
            for i in range(0, self.data_size, bucket_size):
                np.random.shuffle(self.data_indices[i:i + bucket_size])

    def batch(self, batch_type: str, batch_size: int):
        start = time.time()

        logging.info(f"Calling G2SDataset.batch()")
        sys.stdout.flush()

        self.batch_sizes = []

        if batch_type == "samples":
            raise NotImplementedError

        elif batch_type == "atoms":
            raise NotImplementedError

        elif batch_type.startswith("tokens"):
            sample_size = 0
            max_batch_src_len = 0
            max_batch_tgt_len = 0

            for data_idx in self.data_indices:
                src_len = self.src_lens[data_idx]
                tgt_len = self.tgt_lens[data_idx]

                max_batch_src_len = max(src_len, max_batch_src_len)
                max_batch_tgt_len = max(tgt_len, max_batch_tgt_len)
                while self.args.enable_amp and not max_batch_src_len % 8 == 0:  # for amp
                    max_batch_src_len += 1
                while self.args.enable_amp and not max_batch_tgt_len % 8 == 0:  # for amp
                    max_batch_tgt_len += 1

                if batch_type == "tokens" and \
                        max_batch_src_len * (sample_size + 1) <= batch_size:
                    sample_size += 1
                elif batch_type == "tokens_sum" and \
                        (max_batch_src_len + max_batch_tgt_len) * (sample_size + 1) <= batch_size:
                    sample_size += 1
                elif self.args.enable_amp and not sample_size % 8 == 0:
                    sample_size += 1
                else:
                    self.batch_sizes.append(sample_size)

                    sample_size = 1
                    max_batch_src_len = src_len
                    max_batch_tgt_len = tgt_len
                    while self.args.enable_amp and not max_batch_src_len % 8 == 0:  # for amp
                        max_batch_src_len += 1
                    while self.args.enable_amp and not max_batch_tgt_len % 8 == 0:  # for amp
                        max_batch_tgt_len += 1

            '''
            sample_size = 0
            max_batch_src_len = 0

            for data_idx in self.data_indices:
                src_len = self.src_lens[data_idx]
                max_batch_src_len = max(src_len, max_batch_src_len)
                while self.args.enable_amp and not max_batch_src_len % 8 == 0:          # for amp
                    max_batch_src_len += 1

                if max_batch_src_len * (sample_size + 1) <= batch_size:
                    sample_size += 1
                elif self.args.enable_amp and not sample_size % 8 == 0:
                    sample_size += 1
                else:
                    self.batch_sizes.append(sample_size)

                    sample_size = 1
                    max_batch_src_len = src_len
                    while self.args.enable_amp and not max_batch_src_len % 8 == 0:      # for amp
                        max_batch_src_len += 1
            '''

            # lastly
            self.batch_sizes.append(sample_size)
            self.batch_sizes = np.array(self.batch_sizes)
            assert np.sum(self.batch_sizes) == self.data_size, \
                f"Size mismatch! Data size: {self.data_size}, sum batch sizes: {np.sum(self.batch_sizes)}"

            self.batch_ends = np.cumsum(self.batch_sizes)
            self.batch_starts = np.concatenate([[0], self.batch_ends[:-1]])

        else:
            raise ValueError(f"batch_type {batch_type} not supported!")

        logging.info(f"Done, time: {time.time() - start: .2f} s, total batches: {self.__len__()}")
        sys.stdout.flush()

    def __getitem__(self, index: int) -> G2SBatch:
        batch_index = index
        batch_start = self.batch_starts[batch_index]
        batch_end = self.batch_ends[batch_index]

        data_indices = self.data_indices[batch_start:batch_end]

        # collating, essentially
        # source (graph)
        graph_features = []
        atom_size = []
        unimol_data = []
        for data_index in data_indices:
            graph = self.graphs[data_index]
            graph_features.append(graph)
            atom_size.append(graph.num_nodes())

        batch_graph = dgl.batch(graphs=graph_features)
        batch_input = None

        # target (seq)
        tgt_token_ids = self.tgt_token_ids[data_indices]
        tgt_lengths = self.tgt_lens[data_indices]

        tgt_token_ids = tgt_token_ids[:, :max(tgt_lengths)]

        tgt_token_ids = torch.as_tensor(tgt_token_ids, dtype=torch.long)
        tgt_lengths = torch.tensor(tgt_lengths, dtype=torch.long)

        distances = None
        if self.args.compute_graph_distance:
            distances = collate_graph_distances(self.args, graph_features, atom_size)

        """
        logging.info("--------------------src_tokens--------------------")
        for data_index in data_indices:
            smi = "".join(self.vocab_tokens[src_token_id] for src_token_id in self.src_token_ids[data_index])
            logging.info(smi)
        logging.info("--------------------distances--------------------")
        logging.info(f"{distances}")
        exit(0)
        """

        g2s_batch = G2SBatch(
            atom_size=atom_size,
            batch_graph=batch_graph,
            tgt_token_ids=tgt_token_ids,
            tgt_lengths=tgt_lengths,
            batch_input= batch_input,
            distances=distances
        )
        # g2s_batch.log_tensor_shape()

        return g2s_batch

    def __len__(self):
        return len(self.batch_sizes)


def get_graph_features_from_smis(src_lines):
    smis = []
    for i, line in enumerate(src_lines):
        smi = "".join(line.split())
        smis.append(smi)
    
    graphs = []

    for i, smi in enumerate(smis):
        try:
            if not smi.strip():
                smi = "CC"  # hardcode to ignore
                graphs.append(None)
                print("Failed to Get smiles.")
                continue

            if smi == "CC":
                print("Ignor hard code smiles.")
                graphs.append(None)
                continue

            mol = Chem.MolFromSmiles(smi)

            graph = mol_to_bigraph(mol, node_featurizer=node_featurizer_egnn,
                               edge_featurizer=edge_featurizer_egnn)
            AllChem.EmbedMolecule(mol, randomSeed=1)
            AllChem.MMFFOptimizeMolecule(mol)
            coords = get_mol_3d_coordinates(mol)
            
            if coords is None :
                graphs.append(None)
                continue

            coords_tensor = torch.tensor(coords, dtype=torch.float32)

            if torch.all(coords_tensor == 0):
                graphs.append(None)
                continue

            graph.ndata['dist'] = coords_tensor

            graphs.append(graph)

        except Exception as e:
            logging.error(f"Error to get coodination for {smi}")
            graphs.append(None)

        if i > 0 and i % 10000 == 0:
            logging.info(f"Processing {i}th SMILES for Graph Features")
    return graphs


def get_graph_features_from_smi(_args):
    i, smi, use_rxn_class = _args
    try:
        assert isinstance(smi, str) and isinstance(use_rxn_class, bool)
        if i > 0 and i % 10000 == 0:
            logging.info(f"Processing {i}th SMILES")
    
        if not smi.strip():
            smi = "CC"  # hardcode to ignore
        if smi =="CC":
            print("Hard code to ignore")
            return None
        mol = Chem.MolFromSmiles(smi)
        graph = mol_to_bigraph(mol, node_featurizer=node_featurizer_egnn,
                               edge_featurizer=edge_featurizer_egnn)
        AllChem.EmbedMolecule(mol, randomSeed=1)
        AllChem.MMFFOptimizeMolecule(mol)
        coords = get_mol_3d_coordinates(mol)

        if coords is None:
            return None
        coords_tensor = torch.tensor(coords, dtype=torch.float32)
        graph.ndata['dist'] = coords_tensor
        return graph
    except Exception as e:
        logging.error(f"Error to get coodination for {smi}")
        return None


def collate_graph_features(graph_features: List[Tuple], directed: bool = True, use_rxn_class: bool = False) \
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[np.ndarray], List[np.ndarray]]:
    if directed:
        padded_features = get_atom_features_sparse(Chem.Atom("*"), use_rxn_class=use_rxn_class, rxn_class=0)
        fnode = [np.array(padded_features)]
        fmess = [np.zeros(shape=[1, 2 + BOND_FDIM], dtype=np.int32)]
        agraph = [np.zeros(shape=[1, 11], dtype=np.int32)]
        bgraph = [np.zeros(shape=[1, 11], dtype=np.int32)]

        n_unique_bonds = 1
        edge_offset = 1

        atom_scope, bond_scope = [], []

        for bid, graph_feature in enumerate(graph_features):
            a_scope, b_scope, atom_features, bond_features, a_graph, b_graph = graph_feature

            a_scope = a_scope.copy()
            b_scope = b_scope.copy()
            atom_features = atom_features.copy()
            bond_features = bond_features.copy()
            a_graph = a_graph.copy()
            b_graph = b_graph.copy()

            atom_offset = len(fnode)
            bond_offset = n_unique_bonds
            n_unique_bonds += int(bond_features.shape[0] / 2)  # This should be correct?

            a_scope[:, 0] += atom_offset
            b_scope[:, 0] += bond_offset
            atom_scope.append(a_scope)
            bond_scope.append(b_scope)

            # node iteration is reduced to an extend
            fnode.extend(atom_features)

            # edge iteration is reduced to an append
            bond_features[:, :2] += atom_offset
            fmess.append(bond_features)

            a_graph += edge_offset
            a_graph[a_graph >= 999999999] = 0  # resetting padding edge to point towards edge 0
            agraph.append(a_graph)

            b_graph += edge_offset
            b_graph[b_graph >= 999999999] = 0  # resetting padding edge to point towards edge 0
            bgraph.append(b_graph)

            edge_offset += bond_features.shape[0]

        # densification
        fnode = np.stack(fnode, axis=0)
        fnode_one_hot = np.zeros([fnode.shape[0], sum(ATOM_FDIM)], dtype=np.float32)

        for i in range(len(ATOM_FDIM) - 1):
            fnode[:, i + 1:] += ATOM_FDIM[i]  # cumsum, essentially

        for i, feat in enumerate(fnode):  # Looks vectorizable?
            # fnode_one_hot[i, feat[feat < 9999]] = 1
            fnode_one_hot[i, feat[feat < sum(ATOM_FDIM)]] = 1

        fnode = torch.as_tensor(fnode_one_hot, dtype=torch.float)
        fmess = torch.as_tensor(np.concatenate(fmess, axis=0), dtype=torch.float)

        agraph = np.concatenate(agraph, axis=0)
        column_idx = np.argwhere(np.all(agraph[..., :] == 0, axis=0))
        agraph = agraph[:, :column_idx[0, 0] + 1]  # drop trailing columns of 0, leaving only 1 last column of 0

        bgraph = np.concatenate(bgraph, axis=0)
        column_idx = np.argwhere(np.all(bgraph[..., :] == 0, axis=0))
        bgraph = bgraph[:, :column_idx[0, 0] + 1]  # drop trailing columns of 0, leaving only 1 last column of 0

        agraph = torch.as_tensor(agraph, dtype=torch.long)
        bgraph = torch.as_tensor(bgraph, dtype=torch.long)

    else:
        raise NotImplementedError

    return fnode, fmess, agraph, bgraph, atom_scope, bond_scope


def collate_graph_distances(args, graph_features: List[dgl.DGLGraph], a_lengths: List[int]) -> torch.Tensor:
    max_len = max(a_lengths)

    distances = []
    for bid, (graph_feature, a_length) in enumerate(zip(graph_features, a_lengths)):
        bondes = graph_feature.edges()
        bond_features = zip(*bondes)

        # compute adjacency
        adjacency = np.zeros((a_length, a_length), dtype=np.int32)
        for u, v in bond_features:
            u_index = u.item()
            v_index = v.item()
            adjacency[u_index, v_index] = 1

        # compute graph distance
        distance = adjacency.copy()
        shortest_paths = adjacency.copy()
        path_length = 2
        stop_counter = 0
        non_zeros = 0

        while 0 in distance:
            shortest_paths = np.matmul(shortest_paths, adjacency)
            shortest_paths = path_length * (shortest_paths > 0)
            new_distance = distance + (distance == 0) * shortest_paths

            # if np.count_nonzero(new_distance) == np.count_nonzero(distance):
            if np.count_nonzero(new_distance) <= non_zeros:
                stop_counter += 1
            else:
                non_zeros = np.count_nonzero(new_distance)
                stop_counter = 0

            if args.task == "reaction_prediction" and stop_counter == 3:
                break

            distance = new_distance
            path_length += 1

        # bucket
        distance[(distance > 8) & (distance < 15)] = 8
        distance[distance >= 15] = 9
        if args.task == "reaction_prediction":
            distance[distance == 0] = 10

        # reset diagonal
        np.fill_diagonal(distance, 0)

        # padding
        if args.task == "reaction_prediction":
            padded_distance = np.ones((max_len, max_len), dtype=np.int32) * 11
        else:
            padded_distance = np.ones((max_len, max_len), dtype=np.int32) * 10
        padded_distance[:a_length, :a_length] = distance

        distances.append(padded_distance)

    distances = np.stack(distances)
    distances = torch.as_tensor(distances, dtype=torch.long)

    return distances


def make_vocab(fns: Dict[str, List[Tuple[str, str]]], vocab_file: str, tokenized=True):
    assert tokenized, f"Vocab can only be made from tokenized files"

    logging.info(f"Making vocab from {fns}")
    vocab = {}

    for phase, file_list in fns.items():
        for src_file, tgt_file in file_list:
            for fn in [src_file, tgt_file]:
                with open(fn, "r") as f:
                    for line in f:
                        tokens = line.strip().split()
                        for token in tokens:
                            if token in vocab:
                                vocab[token] += 1
                            else:
                                vocab[token] = 1

    logging.info(f"Saving vocab into {vocab_file}")
    with open(vocab_file, "w") as of:
        of.write("_PAD\n_UNK\n_SOS\n_EOS\n")
        for token, count in vocab.items():
            of.write(f"{token}\t{count}\n")


def load_vocab(vocab_file: str) -> Dict[str, int]:
    if os.path.exists(vocab_file):
        logging.info(f"Loading vocab from {vocab_file}")
    else:
        vocab_file = "./preprocessed/default_vocab_smiles.txt"
        logging.info(f"Vocab file invalid, loading default vocab from {vocab_file}")

    vocab = {}
    with open(vocab_file, "r") as f:
        for i, line in enumerate(f):
            token = line.strip().split("\t")[0]
            vocab[token] = i

    return vocab


def data_util_test():
    pass


if __name__ == "__main__":
    data_util_test()
