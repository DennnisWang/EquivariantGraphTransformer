import torch
from dgllife.data.uspto import default_node_featurizer_center, default_edge_featurizer_center, atom_types
from dgllife.utils import mol_to_bigraph, get_mol_3d_coordinates
from dgllife.utils.featurizers import *
from rdkit import Chem
from rdkit.Chem import AllChem

from utils.data_utils import get_graph_features_from_smis

from functools import partial

node_featurizer_egnn = BaseAtomFeaturizer({
    'hv': ConcatFeaturizer(
        [partial(atom_type_one_hot,
                 allowable_set=atom_types, encode_unknown=True),
         partial(atom_degree_one_hot, encode_unknown=True),
         partial(atom_formal_charge_one_hot, encode_unknown=True),
         partial(atom_explicit_valence_one_hot, encode_unknown=True),
         partial(atom_hybridization_one_hot, encode_unknown=True),
         partial(atom_total_num_H_one_hot, encode_unknown=True),
         partial(atom_chirality_type_one_hot, encode_unknown=True),
         partial(atom_num_radical_electrons_one_hot, encode_unknown=True),
         partial(atom_chiral_tag_one_hot, encode_unknown=True),
         partial(atom_implicit_valence_one_hot, encode_unknown=True),
         partial(atom_is_aromatic_one_hot, encode_unknown=True),
         atom_is_aromatic]
    )
})

edge_featurizer_egnn = BaseBondFeaturizer({
    'he': ConcatFeaturizer([
        bond_type_one_hot, bond_is_conjugated_one_hot, bond_is_in_ring_one_hot,bond_stereo_one_hot,bond_direction_one_hot]
    )
})



def test():
    smis = []

    smis.append("[CH3:21][C:22](=[O:23])[OH:24].[CH3:25][OH:26].[CH3:3][CH:4]([CH:5]=[O:6])[CH2:7][c:8]1[cH:9][cH:10][cH:11][cH:12][cH:13]1.[CH:14]([CH2:15][CH3:16])=[O:17].[Cl-:19].[Na+:18].[Na+:2].[OH-:1].[OH2:20]")

    graphs = []

    for i, smi in enumerate(smis):
        try:
            if not smi.strip():
                smi = "CC"  # hardcode to ignore
            mol = Chem.MolFromSmiles(smi)

            # graph = mol_to_bigraph(mol, node_featurizer=default_node_featurizer_center,
            #                        edge_featurizer=default_edge_featurizer_center)
            graph = mol_to_bigraph(mol, node_featurizer=node_featurizer_egnn,
                                   edge_featurizer=edge_featurizer_egnn)
            AllChem.EmbedMolecule(mol, randomSeed=1)
            AllChem.MMFFOptimizeMolecule(mol)
            coords = get_mol_3d_coordinates(mol)

            if coords is None:
                graphs.append(None)
                print(f"Failed To Get 3D Coordination smiles: {smis[i]}")

            coords_tensor = torch.tensor(coords, dtype=torch.float32)
            graph.ndata['dist'] = coords_tensor

            graphs.append(graph)

        except Exception as e:

            graphs.append(None)


    return graphs


if __name__ == '__main__':


    test()