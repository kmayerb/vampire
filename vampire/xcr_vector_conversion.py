"""
Encoding TCRs as one-hot vectors and back again.
Someday this may do the same for BCRs.

The current gene names are for Adaptive data.

Erick: try to keep everything that assumes this gene set in here so
generalization is easier.
"""

from collections import OrderedDict
import pkg_resources

import numpy as np
import pandas as pd
from functools import partial

import vampire.germline_cdr3_aa_tensor as cdr3_tensor

# ### Amino Acids ###

# CDR3Length layer depends on this set and ordering of states.
# So does tcregex.
AA_ORDER = 'ACDEFGHIKLMNPQRSTVWY-'
AA_LIST = list(AA_ORDER)
AA_DICT = {c: i for i, c in enumerate(AA_LIST)}
AA_DICT_REV = {i: c for i, c in enumerate(AA_LIST)}
AA_SET = set(AA_LIST)
AA_NONGAP = [float(c != '-') for c in AA_LIST]

### CHAIN

import os
import json
  
def of_json_file(fname):
    with open(fname, 'r') as fp:
        return json.load(fp)

# THIS BLOCK WAS ADDED TO PROVIDE CONTROL OF CHAIN USAGE, 
# RIGHT NOW THIS IS CONTROLED WITH A SEPARATE FILE.
# IT IS MY INTENT THAT CHAIN COULD BE SPECIFIED IN THE model_spec.json file
path_to_vampire = os.path.dirname(os.path.realpath(__file__))
fname_spec = os.path.join(  path_to_vampire, 
                            'multichain_support',
                            'multichain_spec.json')
if os.path.isfile(fname_spec):
    CHAIN = of_json_file(fname = fname_spec)['chain']
else:
    CHAIN = "beta"

def seq_to_onehot(seq):
    v = np.zeros((len(seq), len(AA_SET)))
    for i, a in enumerate(seq):
        v[i][AA_DICT[a]] = 1
    return v

def onehot_to_seq(onehot):
    return ''.join([AA_DICT_REV[v.argmax()] for v in onehot])


# ### BETA: TCRB ###
# Let's just do a quick test to see if we can override the gene names. 
# V genes:
TCRB_V_GENE_LIST = [
    'TCRBV01-01', 'TCRBV02-01', 'TCRBV03-01', 'TCRBV03-02', 'TCRBV04-01', 'TCRBV04-02', 'TCRBV04-03', 'TCRBV05-01',
    'TCRBV05-02', 'TCRBV05-03', 'TCRBV05-04', 'TCRBV05-05', 'TCRBV05-06', 'TCRBV05-07', 'TCRBV05-08', 'TCRBV06-01',
    'TCRBV06-04', 'TCRBV06-05', 'TCRBV06-06', 'TCRBV06-07', 'TCRBV06-08', 'TCRBV06-09', 'TCRBV07-01', 'TCRBV07-02',
    'TCRBV07-03', 'TCRBV07-04', 'TCRBV07-05', 'TCRBV07-06', 'TCRBV07-07', 'TCRBV07-08', 'TCRBV07-09', 'TCRBV08-02',
    'TCRBV09-01', 'TCRBV10-01', 'TCRBV10-02', 'TCRBV10-03', 'TCRBV11-01', 'TCRBV11-02', 'TCRBV11-03', 'TCRBV12-01',
    'TCRBV12-02', 'TCRBV12-05', 'TCRBV13-01', 'TCRBV14-01', 'TCRBV15-01', 'TCRBV16-01', 'TCRBV18-01', 'TCRBV19-01',
    'TCRBV20-01', 'TCRBV21-01', 'TCRBV22-01', 'TCRBV23-01', 'TCRBV23-or09_02', 'TCRBV25-01', 'TCRBV27-01', 'TCRBV28-01',
    'TCRBV29-01', 'TCRBV30-01', 'TCRBVA-or09_02']
TCRB_V_GENE_LIST = ['TCRDV01-01','TCRDV02-01','TCRDV03-01'] # THIS IS A HACK TEST
TCRB_V_GENE_DICT = {c: i for i, c in enumerate(TCRB_V_GENE_LIST)}
TCRB_V_GENE_DICT_REV = {i: c for i, c in enumerate(TCRB_V_GENE_LIST)}
TCRB_V_GENE_SET = set(TCRB_V_GENE_LIST)

# ### DELTA: TCRDV ###
    # TRDV1*01
    # TRDV2*01
    # TRDV2*02
    # TRDV2*03
    # TRDV3*01
    # TRDV3*02
TCRD_V_GENE_LIST = ['TCRDV01-01','TCRDV02-01','TCRDV03-01']
TCRD_V_GENE_DICT = {c: i for i, c in enumerate(TCRD_V_GENE_LIST)}
TCRD_V_GENE_DICT_REV = {i: c for i, c in enumerate(TCRD_V_GENE_LIST)}
TCRD_V_GENE_SET = set(TCRD_V_GENE_LIST)

#### GAMMA: TCRGV ####
    # TRGV1*01
    # TRGV10*01
    # TRGV10*02
    # TRGV11*01
    # TRGV11*02
    # TRGV2*01
    # TRGV2*02
    # TRGV2*03
    # TRGV3*01
    # TRGV3*02
    # TRGV4*01
    # TRGV4*02
    # TRGV5*01
    # TRGV5P*01
    # TRGV5P*02
    # TRGV8*01
    # TRGV9*01
    # TRGV9*02
    # TRGVA*01
TCRG_V_GENE_LIST = ["TCRGV01-01","TCRGV10-01","TCRGV10-02","TCRGV11-01",
                    "TCRGV11-02","TCRGV02-01","TCRGV02-02","TCRGV02-03",
                    "TCRGV03-01","TCRGV03-02","TCRGV04-01","TCRGV04-02",
                    "TCRGV05-01","TCRGV05P-01","TCRGV05P-02","TCRGV08-01",
                    "TCRGV09-01","TCRGV09-02","TCRGVA-01"]

TCRG_V_GENE_DICT = {c: i for i, c in enumerate(TCRG_V_GENE_LIST)}
TCRG_V_GENE_DICT_REV = {i: c for i, c in enumerate(TCRG_V_GENE_LIST)}
TCRG_V_GENE_SET = set(TCRG_V_GENE_LIST)


# J genes:
# BETA - J genes

TCRB_J_GENE_LIST = [
    'TCRBJ01-01', 'TCRBJ01-02', 'TCRBJ01-03', 'TCRBJ01-04', 'TCRBJ01-05', 'TCRBJ01-06', 'TCRBJ02-01', 'TCRBJ02-02',
    'TCRBJ02-03', 'TCRBJ02-04', 'TCRBJ02-05', 'TCRBJ02-06', 'TCRBJ02-07']
TCRB_J_GENE_LIST = ['TCRDJ01-01', 'TCRDJ02-01','TCRDJ03-01','TCRDJ04-01'] # This is a temporary hack test
TCRB_J_GENE_DICT = {c: i for i, c in enumerate(TCRB_J_GENE_LIST)}
TCRB_J_GENE_DICT_REV = {i: c for i, c in enumerate(TCRB_J_GENE_LIST)}
TCRB_J_GENE_SET = set(TCRB_J_GENE_LIST)

# DELTA - J gene
TCRD_J_GENE_LIST = ['TCRDJ01-01', 'TCRDJ02-01','TCRDJ03-01','TCRDJ04-01']

TCRD_J_GENE_DICT = {c: i for i, c in enumerate(TCRD_J_GENE_LIST)}
TCRD_J_GENE_DICT_REV = {i: c for i, c in enumerate(TCRD_J_GENE_LIST)}
TCRD_J_GENE_SET = set(TCRD_J_GENE_LIST)


# GAMMA - J gene
TCRG_J_GENE_LIST = ["TCRGJ01-01","TCRGJ01-02","TCRGJ02-01","TCRGJP-01","TCRGJP1-01","TCRGJP2-01"]

TCRG_J_GENE_DICT = {c: i for i, c in enumerate(TCRG_J_GENE_LIST)}
TCRG_J_GENE_DICT_REV = {i: c for i, c in enumerate(TCRG_J_GENE_LIST)}
TCRG_J_GENE_SET = set(TCRG_J_GENE_LIST)


for gene in TCRB_V_GENE_LIST + TCRB_J_GENE_LIST:
    # We need to specify how long strings are for numpy initialization. See onehot_to_padded_tcrbs.
    assert len(gene) < 20


def vgene_to_onehot_general(v_gene, 
                    gene_set = TCRB_V_GENE_SET, 
                    chain_v_gene_dict=TCRB_V_GENE_DICT):
    v = np.zeros(len(gene_set))
    v[chain_v_gene_dict[v_gene]] = 1
    return v

def onehot_to_vgene_general(onehot, 
                    chain_v_gene_dict_rev=TCRB_V_GENE_DICT_REV):
    return chain_v_gene_dict_rev[onehot.argmax()]

def jgene_to_onehot_general(j_gene, 
                    gene_set = TCRB_J_GENE_SET, 
                    chain_j_gene_dict=TCRB_J_GENE_DICT):
    v = np.zeros(len(gene_set))
    v[chain_j_gene_dict[j_gene]] = 1
    return v

def onehot_to_jgene_general(onehot,
                    chain_j_gene_dict_rev=TCRB_J_GENE_DICT_REV):
    return chain_j_gene_dict_rev[onehot.argmax()]

if CHAIN == "beta":
    vgene_to_onehot = partial(vgene_to_onehot_general, gene_set = TCRB_V_GENE_SET, chain_v_gene_dict=TCRB_V_GENE_DICT)
    onehot_to_vgene = partial(onehot_to_vgene_general, chain_v_gene_dict_rev=TCRB_V_GENE_DICT_REV)
    jgene_to_onehot = partial(jgene_to_onehot_general, gene_set = TCRB_J_GENE_SET, chain_j_gene_dict=TCRB_J_GENE_DICT)
    onehot_to_jgene = partial(onehot_to_jgene_general, chain_j_gene_dict_rev=TCRB_J_GENE_DICT_REV)
if CHAIN == "delta":
    vgene_to_onehot = partial(vgene_to_onehot_general, gene_set = TCRD_V_GENE_SET, chain_v_gene_dict=TCRD_V_GENE_DICT)
    onehot_to_vgene = partial(onehot_to_vgene_general, chain_v_gene_dict_rev=TCRD_V_GENE_DICT_REV)
    jgene_to_onehot = partial(jgene_to_onehot_general, gene_set = TCRD_J_GENE_SET, chain_j_gene_dict=TCRD_J_GENE_DICT)
    onehot_to_jgene = partial(onehot_to_jgene_general, chain_j_gene_dict_rev=TCRD_J_GENE_DICT_REV)
if CHAIN == "gamma":
    vgene_to_onehot = partial(vgene_to_onehot_general, gene_set = TCRG_V_GENE_SET, chain_v_gene_dict=TCRG_V_GENE_DICT)
    onehot_to_vgene = partial(onehot_to_vgene_general, chain_v_gene_dict_rev=TCRG_V_GENE_DICT_REV)
    jgene_to_onehot = partial(jgene_to_onehot_general, gene_set = TCRG_J_GENE_SET, chain_j_gene_dict=TCRG_J_GENE_DICT)
    onehot_to_jgene = partial(onehot_to_jgene_general, chain_j_gene_dict_rev=TCRG_J_GENE_DICT_REV)

# The goal here is to retain original behavior of the 4 function, but allow change based on CHAIN arg, using functools
# Therefore if chain is "beta" exisitng tests should pass
#21:24 $ pytest vampire/tests/test_xcr_vector_conversion.py
#================================================================================================================== test session starts ==================================================================================================================
# vampire/tests/test_xcr_vector_conversion.py::test_pad_middle PASSED                                                                                                                                                                               [ 16%]
# vampire/tests/test_xcr_vector_conversion.py::test_gene_conversion PASSED                                                                                                                                                                          [ 33%]
# vampire/tests/test_xcr_vector_conversion.py::test_aa_conversion PASSED                                                                                                                                                                            [ 50%]
# vampire/tests/test_xcr_vector_conversion.py::test_cdr3_length_of_onehots PASSED                                                                                                                                                                   [ 66%]
# vampire/tests/test_xcr_vector_conversion.py::test_contiguous_match_counts PASSED                                                                                                                                                                  [ 83%]
# vampire/tests/test_xcr_vector_conversion.py::test_contiguous_match_counts_df PASSED                                                                                                                                                               [100%]  
# When we try CHAIN = "delta"

# vampire/tests/test_xcr_vector_conversion.py::test_gene_conversion FAILED                                                                                                                                                                          [ 33%]
# vampire/tests/test_xcr_vector_conversion.py::test_cdr3_length_of_onehots FAILED                                                                                                                                                                   [ 66%]



def pad_middle(seq, desired_length):
    """
    Pad the middle of a sequence with gaps so that it is a desired length.
    Fail assertion if it's already longer than `desired_length`.
    """
    seq_len = len(seq)
    assert seq_len <= desired_length
    pad_start = seq_len // 2
    pad_len = desired_length - seq_len
    return seq[:pad_start] + '-' * pad_len + seq[pad_start:]


def unpad(seq):
    """
    Remove gap padding.
    """
    return seq.translate(seq.maketrans('', '', '-'))


def avj_triple_to_tcr_df(amino_acid, v_gene, j_gene):
    """
    Put our TCR triple into an appropriate DataFrame.
    """
    return pd.DataFrame(OrderedDict([('amino_acid', amino_acid), ('v_gene', v_gene), ('j_gene', j_gene)]))


def avj_raw_triple_to_tcr_df(amino_acid, v_gene, j_gene):
    """
    A "raw" triple here means as a big np array.
    """
    return avj_triple_to_tcr_df([amino_acid[i, :, :] for i in range(amino_acid.shape[0])],
                                [v_gene[i, :] for i in range(v_gene.shape[0])],
                                [j_gene[i, :] for i in range(j_gene.shape[0])])


def unpadded_tcrbs_to_onehot(df, desired_length):
    """
    Translate a data frame of TCR betas written as (CDR3 sequence, V gene name,
    J gene name) into onehot-encoded format with CDR3 padding out to
    `desired_length`.
    If a CDR3 sequence exceeds `desired_length` this will fail an assertion.
    """

    return avj_triple_to_tcr_df(
        df['amino_acid'].apply(lambda s: seq_to_onehot(pad_middle(s, desired_length))),
        df['v_gene'].apply(vgene_to_onehot),
        df['j_gene'].apply(jgene_to_onehot)
        )  # yapf: disable


def onehot_to_padded_tcrbs(amino_acid_arr, v_gene_arr, j_gene_arr):
    """
    Convert back from onehot encoding arrays to padded TCR betas.

    Say there are n sequences.

    :param amino_acid_arr: onehot array of shape (n, max_cdr3_len, 21).
    :param v_gene_arr: onehot array of shape (n, n_v_genes).
    :param j_gene_arr: onehot array of shape (n, n_j_genes).
    """

    def aux(f, a):
        """
        Convert 2D numpy array to an array of strings by applying f to every row.
        It's like np.apply_along_axis(f, 1, a) but without this truncation problem:
        https://github.com/numpy/numpy/issues/8352

        Note that we assume the strings are of maximum length 20. We check that the
        gene names satisfy this with an assert above.
        """
        nrows = a.shape[0]
        # Here's where the length-20 assumption lives.
        out = np.empty((nrows, ), dtype=np.dtype('<U20'))
        for i in range(nrows):
            out[i] = f(a[i])
        return out

    return avj_triple_to_tcr_df(
        np.array([onehot_to_seq(amino_acid_arr[i]) for i in range(amino_acid_arr.shape[0])]),
        aux(onehot_to_vgene, v_gene_arr),
        aux(onehot_to_jgene, j_gene_arr)
        )  # yapf: disable


def onehot_to_tcrbs(amino_acid_arr, v_gene_arr, j_gene_arr):
    """
    Convert back from onehot encodings to TCR betas.
    """

    df = onehot_to_padded_tcrbs(amino_acid_arr, v_gene_arr, j_gene_arr)
    return avj_triple_to_tcr_df(df['amino_acid'].apply(unpad), df['v_gene'], df['j_gene'])


def adaptive_aa_encoding_tensors(max_cdr3_len):
    germline_cdr3_csv = pkg_resources.resource_filename('vampire', 'data/germline-cdr3-aas.csv')

    return cdr3_tensor.aa_encoding_tensors(germline_cdr3_csv, AA_ORDER, TCRB_V_GENE_LIST, TCRB_J_GENE_LIST,
                                           max_cdr3_len)


def cdr3_length_of_onehots(onehot_cdr3s: pd.Series):
    """
    Compute the CDR3 length of one-hot-encoded CDR3s.

    :param onehot_cdr3s: A Series of numpy one-hot-encoded arrays.

    :return: a float array of CDR3 lengths.
    """
    # We assume that gap is the 21st amino acid.
    all_but_gap_mask = np.array([AA_NONGAP]).transpose()
    return onehot_cdr3s.apply(lambda row: np.sum(row.dot(all_but_gap_mask)))


def contiguous_match_counts(padded_onehot, v_germline_aa_onehot, j_germline_aa_onehot):
    """
    Get a 2-component array representing the number of contiguous matches to
    germline on the V and the J. Here contiguous means without interruption
    from the left of the v and without interruption from the right of the j.

    :param padded_onehot: padded onehot-encoded matrix for the CDR3 amino
    acids
    :param v_germline_aas: onehot-encoded matrix for the V germline-encoded
    amino acids
    :param j_germline_aas: onehot-encoded matrix for the J germline-encoded
    amino acids
    :return: a 2-vector describing the number of contigious states that match
    the supplied v and j germlines

    If the max_cdr3_len is 30, then all of the inputs are (30, 21) and the
    output is length 30.

    See tests for examples.
    """
    return np.array([
        # Here cumprod ensures that as soon as we get a zero, it's zero thereafter.
        np.sum(np.cumprod(np.sum(np.multiply(padded_onehot, v_germline_aa_onehot), axis=1))),
        np.sum(np.cumprod(np.flip(np.sum(np.multiply(padded_onehot, j_germline_aa_onehot), axis=1), axis=0)))
    ])


def contiguous_match_counts_df(onehot_df, v_germline_aa_tensor, j_germline_aa_tensor):
    """
    Apply contiguous_match_counts appropriately to every row of a
    onehot-encoded data frame.

    :param onehot_df: a dataframe like one gets from unpadded_tcrbs_to_onehot
    :param v_germline_aa_tensor: a tensor like from adaptive_aa_encoding_tensors
    :param j_germline_aa_tensor: a tensor like from adaptive_aa_encoding_tensors
    :return: a numpy array of shape (2, len(onehot_df)) giving the v and j
    contiguous match count for every input.
    """
    return np.vstack(
        onehot_df.apply(
            lambda row: contiguous_match_counts(
                row['amino_acid'],
                # The following two lines obtain the germline aas for the germline
                # calls of the row.
                np.tensordot(row['v_gene'], v_germline_aa_tensor, axes=1),
                np.tensordot(row['j_gene'], j_germline_aa_tensor, axes=1)),
            axis=1))
