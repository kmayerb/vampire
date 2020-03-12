import numpy as np
import pytest
import json
import vampire.common as common
import vampire.xcr_vector_conversion as conversion
import os


def of_json_file(fname):
    with open(fname, 'r') as fp:
        return json.load(fp)

# THIS BLOCK WAS ADDED TO PROVIDE CONTROL OF CHAIN USAGE, 
# RIGHT NOW THIS IS CONTROLED WITH A SEPARATE FILE.
# IT IS MY INTENT THAT CHAIN COULD BE SPECIFIED IN THE model_spec.json file
path_to_test= os.path.dirname(os.path.realpath(__file__))
path_to_vampire = os.path.dirname(path_to_test)
fname_spec = os.path.join(  path_to_vampire, 
                            'multichain_support',
                            'multichain_spec.json')
if os.path.isfile(fname_spec):
    CHAIN = of_json_file(fname = fname_spec)['chain']
else:
    CHAIN = "beta"


@pytest.mark.skipif(CHAIN != "beta",
                    reason="This test requires beta chain mode")
def test_gene_conversion():
    for gene in conversion.TCRB_V_GENE_LIST:
        assert conversion.onehot_to_vgene(conversion.vgene_to_onehot(gene)) == gene
    for gene in conversion.TCRB_J_GENE_LIST:
        assert conversion.onehot_to_jgene(conversion.jgene_to_onehot(gene)) == gene

@pytest.mark.skipif(CHAIN != "delta",
                    reason="This test requires delta chain mode")
def test_gene_conversion_delta():
    for gene in conversion.TCRD_V_GENE_LIST:
        assert conversion.onehot_to_vgene(conversion.vgene_to_onehot(gene)) == gene
    for gene in conversion.TCRD_J_GENE_LIST:
        assert conversion.onehot_to_jgene(conversion.jgene_to_onehot(gene)) == gene

@pytest.mark.skipif(CHAIN != "gamma",
                    reason="This test requires gamma chain mode")
def test_gene_conversion_gamma():
    for gene in conversion.TCRG_V_GENE_LIST:
        assert conversion.onehot_to_vgene(conversion.vgene_to_onehot(gene)) == gene
    for gene in conversion.TCRG_J_GENE_LIST:
        assert conversion.onehot_to_jgene(conversion.jgene_to_onehot(gene)) == gene

@pytest.mark.skipif(CHAIN != "beta",
                    reason="This test requires beta chain mode")
def test_cdr3_length_of_onehots():
    data = common.read_data_csv('adaptive-filter-test.correct.csv')
    lengths = data['amino_acid'].apply(len).apply(float)
    onehots = conversion.unpadded_tcrbs_to_onehot(data, 30)
    assert lengths.equals(conversion.cdr3_length_of_onehots(onehots['amino_acid']))

@pytest.mark.skipif(CHAIN != "delta",
                    reason="This test requires delta chain mode")
def test_cdr3_length_of_onehots_delta():
    data = common.read_data_csv('delta_1000_test.csv')
    lengths = data['amino_acid'].apply(len).apply(float)
    onehots = conversion.unpadded_tcrbs_to_onehot(data, 30)
    assert lengths.equals(conversion.cdr3_length_of_onehots(onehots['amino_acid']))

@pytest.mark.skipif(CHAIN != "gamma",
                    reason="This test requires gamma chain mode")
def test_cdr3_length_of_onehots_gamma():
    data = common.read_data_csv('gamma_1000_test.csv')
    lengths = data['amino_acid'].apply(len).apply(float)
    onehots = conversion.unpadded_tcrbs_to_onehot(data, 30)
    assert lengths.equals(conversion.cdr3_length_of_onehots(onehots['amino_acid']))

@pytest.mark.skipif(CHAIN != "beta",
                reason="This test requires beta chain mode")   
def test_contiguous_match_counts_df():
    test = conversion.unpadded_tcrbs_to_onehot(common.read_data_csv('adaptive-filter-test.correct.csv'), 30)
    v_germline_tensor, j_germline_tensor = conversion.adaptive_aa_encoding_tensors(30)
    result = conversion.contiguous_match_counts_df(test, v_germline_tensor, j_germline_tensor)

    assert np.array_equal(result[0], np.array([4., 6.]))
    assert np.array_equal(result[1], np.array([5., 5.]))
    assert np.array_equal(result[5], np.array([5., 0.]))

@pytest.mark.skipif(CHAIN != "beta",
                reason="This test requires beta chain mode")   
def test_contiguous_match_counts_df():
    test = conversion.unpadded_tcrbs_to_onehot(common.read_data_csv('adaptive-filter-test.correct.csv'), 30)
    v_germline_tensor, j_germline_tensor = conversion.adaptive_aa_encoding_tensors(30)
    result = conversion.contiguous_match_counts_df(test, v_germline_tensor, j_germline_tensor)

    assert np.array_equal(result[0], np.array([4., 6.]))
    assert np.array_equal(result[1], np.array([5., 5.]))
    assert np.array_equal(result[5], np.array([5., 0.]))

@pytest.mark.skipif(CHAIN != "delta",
                reason="This test requires delta chain mode")   
def test_contiguous_match_counts_df_delta():
    test = conversion.unpadded_tcrbs_to_onehot(common.read_data_csv('delta_1000_test.csv'), 30)
    v_germline_tensor, j_germline_tensor = conversion.adaptive_aa_encoding_tensors(30)
    result = conversion.contiguous_match_counts_df(test, v_germline_tensor, j_germline_tensor)
    
    assert np.array_equal(result[0], np.array([4., 6.]))
    assert np.array_equal(result[1], np.array([5., 5.]))
    assert np.array_equal(result[5], np.array([5., 0.]))


def test_pad_middle():
    with pytest.raises(AssertionError):
        conversion.pad_middle('AAAAA', 2)
    assert 'A---B' == conversion.pad_middle('AB', 5)
    assert 'A--BC' == conversion.pad_middle('ABC', 5)
    assert 'AB' == conversion.pad_middle('AB', 2)
    assert '----' == conversion.pad_middle('---', 4)


def test_aa_conversion():
    target = 'CASY'
    assert conversion.onehot_to_seq(conversion.seq_to_onehot(target)) == target
    target = 'C-SY'
    assert conversion.onehot_to_seq(conversion.seq_to_onehot(target)) == target




def test_contiguous_match_counts():
    v, j = conversion.adaptive_aa_encoding_tensors(30)
    v01_01 = v[0]
    j01_02 = j[1]
    v01_v02_mixture = (v[0] + v[1]) / 2

    v01_j02_matching = conversion.seq_to_onehot('CTSSQ-------------------NYGYTF')
    two_v01_match = conversion.seq_to_onehot('CTWSQ-------------------NYGYTF')
    three_j02_match = conversion.seq_to_onehot('CTSSQ-------------------NYAYTF')

    # Here we have a complete match for both genes.
    assert np.array_equal(conversion.contiguous_match_counts(v01_j02_matching, v01_01, j01_02), np.array([5., 6.]))
    # Here the V match is interrupted by a W instead of an S, and we can see the "contiguous" requirement working.
    assert np.array_equal(conversion.contiguous_match_counts(two_v01_match, v01_01, j01_02), np.array([2., 6.]))
    # Equivalent test for J.
    assert np.array_equal(conversion.contiguous_match_counts(three_j02_match, v01_01, j01_02), np.array([5., 3.]))
    # For the mixture, we have one residue that matches both, then one that
    # only matches V01, then another two that match both. You can see that in
    # the indicator decay from left to right.
    # Here are the indicator vectors before summing:
    # 1., 0.5, 0.5, 0.5, 0.25, 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    # 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1.
    assert np.array_equal(
        conversion.contiguous_match_counts(v01_j02_matching, v01_v02_mixture, j01_02), np.array([2.75, 6.]))

