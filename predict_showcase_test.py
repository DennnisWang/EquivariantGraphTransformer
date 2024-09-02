import argparse
import logging

from models.graph2seq_series_rel import Graph2SeqSeriesRel
from torch.utils.data import DataLoader
from utils.data_utils import canonicalize_smiles, load_vocab, G2SDataset, collate_graph_distances, G2SBatch
from utils import parsing
from utils.train_utils import log_tensor, param_count, set_seed, setup_logger
import torch
from preprocess import get_token_ids, get_graph_features_from_smis
import os
import dgl


class Args():

    def __init__(self):
        self.model = 'g2s_series_rel'
        self.data_name = 'USPTO_STEREO'
        self.task = 'reaction_prediction'
        self.seed = 42
        self.verbose = False
        self.llm = False
        self.new = False
        self.representation_end = ''
        self.log_file = ''
        self.vocab_file = 'D:/data/personal/ai4science/egnn2smiles/preprocessed/USPTO_STEREO_g2s_series_rel_smiles_smiles/vocab_smiles.txt'
        self.preprocess_output_path = ''
        self.save_dir = ''
        self.train_src = ''
        self.train_tgt = ''
        self.val_src = ''
        self.val_tgt = './data/USPTO_STEREO/tgt-val.txt'
        self.test_src = ''
        self.test_tgt = ''
        self.representation_start = ''
        self.do_tokenize = False
        self.make_vocab_only = False
        self.max_seq_len = 512
        self.train_bin = ''
        self.valid_bin = './preprocessed/USPTO_STEREO_g2s_series_rel_smiles_smiles/val_0.npz'
        self.load_from = 'D:/data/personal/ai4science/egnn2smiles/checkpoints/USPTO_STEREO_egnn-1-0.1-11-6-8-256-gcn-8/model.280000_27.pt'
        self.embed_size = 256
        self.share_embeddings = False
        self.mpn_type = 'dgat'
        self.encoder_num_layers = 1
        self.encoder_hidden_size = 256
        self.encoder_attn_heads = 8
        self.encoder_filter_size = 2048
        self.encoder_norm = 'none'
        self.encoder_skip_connection = 'none'
        self.encoder_positional_encoding = 'transformer'
        self.encoder_emb_scale = ''
        self.compute_graph_distance = False
        self.attn_enc_num_layers = 6
        self.attn_enc_filter_size = 2048
        self.attn_enc_hidden_size = 256
        self.attn_enc_heads = 8
        self.rel_pos = 'emb_only'
        self.rel_pos_buckets = 11
        self.decoder_num_layers = 6
        self.decoder_hidden_size = 256
        self.decoder_attn_heads = 8
        self.decoder_filter_size = 2048
        self.dropout = 0.1
        self.attn_dropout = 0.1
        self.max_relative_positions = 4
        self.enable_amp = False
        self.epoch = 300
        self.max_steps = 1000000
        self.warmup_steps = 8000
        self.lr = 4.0
        self.beta1 = 0.9
        self.beta2 = 0.998
        self.eps = 1e-09
        self.weight_decay = 0.0
        self.clip_norm = 20.0
        self.batch_type = 'tokens'
        self.train_batch_size = 4096
        self.valid_batch_size = 4096
        self.accumulation_count = 1
        self.log_iter = 100
        self.eval_iter = 100
        self.save_iter = 100
        self.do_profile = False
        self.record_shapes = False
        self.kernel = ''
        self.do_predict = False
        self.do_score = False
        self.predict_batch_size = '512'
        self.test_bin = ''
        self.result_file = ''
        self.beam_size = 30
        self.n_best = 30
        self.temperature = 0.7
        self.predict_min_len = 1
        self.predict_max_len = 512


def get_features_from_smiles(smile):

    src_tokens = smile.strip().split()
    if not src_tokens:
        src_tokens = ["C", "C"]  # hardcode to ignore

    graph = get_graph_features_from_smis(src_tokens)

    batch_graph = dgl.batch(graphs=graph)

    atom_size = graph[0].num_nodes()

    distances = collate_graph_distances(args, graph, [atom_size])

    return batch_graph, [atom_size], distances


def handle_input(smile):
    graph, atom_size, distances = get_features_from_smiles(smile)

    g2s_batch = G2SBatch(
        atom_size=atom_size,
        batch_graph=graph,
        tgt_token_ids=None,
        tgt_lengths=None,
        batch_input=None,
        distances=distances
    )

    return g2s_batch


def predict_result(args, smiles):
    assert os.path.exists(args.load_from), f"{args.load_from} does not exist!"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    state = torch.load(args.load_from)
    pretrain_args = state["args"]
    pretrain_state_dict = state["state_dict"]

    for attr in ["mpn_type", "rel_pos"]:
        try:
            getattr(pretrain_args, attr)
        except AttributeError:
            setattr(pretrain_args, attr, getattr(args, attr))

    assert args.model == pretrain_args.model, f"Pretrained model is {pretrain_args.model}!"

    model_class = Graph2SeqSeriesRel
    dataset_class = G2SDataset
    args.compute_graph_distance = True

    # initialization ----------------- vocab
    vocab = load_vocab(args.vocab_file)
    vocab_tokens = [k for k, v in sorted(vocab.items(), key=lambda tup: tup[1])]

    model = model_class(pretrain_args, vocab)
    model.load_state_dict(pretrain_state_dict)
    logging.info(f"Loaded pretrained state_dict from {args.load_from}")

    model.to(device)
    model.eval()

    test_batch = handle_input(smiles)
    test_batch.to(device)
    # logging.info(model)
    logging.info(f"Number of parameters = {param_count(model)}")
    with torch.no_grad():
        results = model.predict_step(
            reaction_batch=test_batch,
            batch_size=test_batch.size,
            beam_size=args.beam_size,
            n_best=args.n_best,
            temperature=args.temperature,
            min_length=args.predict_min_len,
            max_length=args.predict_max_len
        )
        smis = []
        for predictions in results["predictions"]:
            for prediction in predictions:
                predicted_idx = prediction.detach().cpu().numpy()
                predicted_tokens = [vocab_tokens[idx] for idx in predicted_idx[:-1]]
                smi = "".join(predicted_tokens)
                smis.append(smi)

        smis_predict = [canonicalize_smiles(smi, trim=False) for smi in smis]

        return smis_predict, smis


if __name__ == '__main__':
    args = Args()
    set_seed(args.seed)
    logger = setup_logger(args, warning_off=True)
    torch.set_printoptions(profile="full")
    michael_addition_6 = 'O=C(C=CCC/1)C1=C\C2=CC=CC=C2.N#CCC#N.N[C@H]([C@H]3[N@](C[C@@H]4C=C)CC[C@H]4C3)C5=CC=NC6=C5C=C(OC)C=C6.OC7=CC=CC=C7C(O)=O' # not right
    michael_addition_5 = 'N#CCC#N.N[C@H]([C@H]1[N@](C[C@@H]2C=C)CC[C@H]2C1)C3=CC=NC4=C3C=C(OC)C=C4.OC5=CC=CC=C5C(O)=O.O=C(C=CC/6)C6=C\C7=CC=CC=C7' # not right
    reactant = 'O=C(C=CCC/1)C1=C\C2=CC=CC=C2.N#CCC#N.C=C[C@@H]3[C@@H]4C[C@H]([N@@](CC4)C3)[C@H](C5=CC=NC6=C5C=C(OC)C=C6)N.OC7=CC=CC=C7C(O)=O'

    # step1 = 'O[C@H]1[C@@]2(C)C(CCC2=O)C3CC=C4C[C@@H](OC(C)=O)CC[C@]4(C)C3C1.C#C[Mg]Br' # right n =1
    # step2 = 'C#CC1(CCC2C3CC=C4CC(CCC4(C3CC(C21C)O)C)OC(C)=O)O.N#C[Zn]C#N.CC(/C=C(O[Ni]O/C(C)=C\C(C)=O)/C)=O.CC5=NC6=C(C=C5)C=CC7=CC=C(C)N=C76.[Mn]'
    # step3 = 'C=C(C#N)C1(CCC2C3CC=C4CC(CCC4(C3CC(C21C)O)C)OC(C)=O)O.O=S(C(F)(F)F)(OS(=O)(C(F)(F)F)=O)=O.ClC5=NC=CC=C5.CS(=O)(O)=O'
    # step4 = 'CC(OC1CC2=CCC(C(CCC(C(C#N)=C)(O)C3C)=C3C4)C4C2(C)CC1)=O.Cl[Rh](P(C5=CC=CC=C5)(C6=CC=CC=C6)C7=CC=CC=C7)(P(C8=CC=CC=C8)(C9=CC=CC=C9)C%10=CC=CC=C%10)P(C%11=CC=CC=C%11)(C%12=CC=CC=C%12)C%13=CC=CC=C%13.[HH]'
    # step6 = 'OC1CC2=CCC(C3=CC=C(C(C)C=O)C(C)=C3C4)C4C2(C)CC1.CCO[Ti](OCC)(OCC)OCC.O=[S@@](C(C)(C)C)N' # not right , n=1 -> CC(C=NS(=O)C(C)(C)C)C1=CCC2C(CC3C2CC=C2CC(O)CCC23C)C1C
    # step6_2 = 'CCO[Ti](OCC)(OCC)OCC.O=[S@@](C(C)(C)C)N.OC1CC2=CCC(C3C(C4)=C(C)C(C(C)C=O)=CC3)C4C2(C)CC1'
    # step7 = 'I[Sm]I.CC(C)(O)C.O=CC[C@H](C)C(OC)=O.OC1CC2=CCC(C3=CC=C(C(C)/C=C/[S@@](C(C)(C)C)=O)C(C)=C3C4)C4C2(C)CC1.[N]' # close, n=2 ->COC(=O)[C@H](C)CC(O)C(C)c1ccc2c(c1C)C[C@H]1[C@@H]2CC=C2C[C@@H](O)CC[C@]21C
    # steps = [step1,step2,step3,step4,step6,step6_2,step7]

    step0 = 'C/C=C/C(OC)=O.CC(C)N(C(C)C)[Li].CN(P(N(C)C)(N(C)C)=O)C'
    step1 = 'ICCCOCC1=CC=C(OC)C=C1.C[Si](C)(Cl)C(C)(C)C.COC(CC(C(C(N)=O)=O)C)=O.[Li].[AlH4]'
    step4 = 'O=C1CC([C@]2([C@@H]3O)C[C@H]1C3=C)[C@]45[C@@H](OC)CC[C@]6(C)[C@H]4C[C@H]2C5N(CC)C6.O=CC(F)(F)F'
    step5 = 'C=C1[C@@H](O)[C@]2(C[C@@H]1C3(OCCO3)C4)C4[C@]56[C@@H](OC)CC[C@]7(C)[C@H]5C[C@H]2C6N(CC)C7.O=[Se]=O.CC(C)(OO)C'

    step6_0 = '[H]C(C1=CC(CCl)=C(O)C=C1)=O.[O][Ca]C([O])=O'
    step6 = '[H]C(C1=CC(CO)=C(O)C=C1)=O.CC(C)(OC)OC.CC2=CC=C(S(=O)(O)=O)C=C2' # right top-1
    step7 = 'BrC1=CC(CO)=C(O)C=C1.CC(C)(OC)OC.CC2=CC=C(S(=O)(O)=O)C=C2' #right top1
    step8 = 'CC1(OCc2c(O1)ccc(Br)c2)C.CCCC[Li].O=CN(C)C' #right top-1
    step9 = 'CC1(OCc2c(O1)ccc(C=O)c2)C.Cl[Cu]Cl.O'
    step9_1 = 'CCN(CC)CC.C[No][No].CC1(OCc2c(O1)ccc(C(Cl)=O)c2)C'
    step10 = 'CC1(OCc2c(O1)ccc([C@@H](O)C[N+]([O-])=O)c2)C.[Pd][C].CO.[HH]' #right top1
    step11 = 'CC1(OCc2c(O1)ccc(C(CN)O)c2)C.BrCCCCCCOCCCCC3=CC=CC=C3.O=CN(C)C' #right top1
    step12 = 'CC1(OCc2c(O1)ccc(C(CNCCCCCCOCCCCc3ccccc3)O)c2)C.CC(O)=O.O' #right top2

    step_a = 'O=C(NC1=CC(C(F)(F)F)=CC=C1)N2C=CC3=C2C=CC(OC4=NC=CC5=C4CC(C)NC5)=C3.BrCC(OC)=O'
    step_b = 'O=C(O[Pd]OC(C)=O)C.CC(P(C(C)(C)C)C(C)(C)C)(C)C.BrC1=CC(C2=C3C=C4C(C(C=CC=C5)=C5C4(C)C)=C2)=C(N3C(C=C6)=CC7=C6C(C=CC=C8)=C8C9=C7C=CC=C9)C=C1.C%10%11=CC=CC=C%10NC%12=C%11C=CC=C%12'
    rxn_72 = 'O=C(OC)C1=C(S(=O)(Cl)=O)C=CC=C1[N+]([O-])=O.NCCN2CCCCC2.C3CCCO3'
    rxn_61 = 'BrC1=CC=C(C(CBr)=O)C=C1.O=C(CC(O)CCCCCCCCCCCC(C)C)OC.[Li]O.O'
    rxn_59 = 'CCC(C(C1=CC=CC=C1)(O)C2=CC=C(N(S(=O)(C)=O)CCN(CC)CC)C=C2)C3=CC=CC=C3.CCO.CC(O)C.Cl'
    rxn_68 = 'OC1=CC=C(C#N)C=C1C(N)=O.CC2(C)N=C(Cl)C3=CC(C#N)=CC=C3O2.CC(C)=O.O=C4NC=CC=C4.O=CN(C)C.N#N.[NaH]'
    rxn_76 = 'OCCP(OCC)(OCC)=O.O=S(OS(=O)(C(F)(F)F)=O)(C(F)(F)F)=O.CCOCC.CC1=NC(C)=CC=C1'
    steps = [step_a,step_b,rxn_72,rxn_61, rxn_59, rxn_76,rxn_68]

    for i,step in enumerate(steps):
        predict_product, smis = predict_result(args, step)
        print("step: ",i)
        print(predict_product)
        print(smis)