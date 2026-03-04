# a variety of utility functions imported by many of the scripts in this repo

from Bio.Seq import Seq
import numpy as np
import os
import pandas as pd
from Levenshtein import distance
from anarci import anarci
import subprocess
from glob import glob
from sklearn.cluster import AgglomerativeClustering
from functools import partial
import yaml
from tqdm import tqdm
from itertools import repeat

#get main dir for repo
repo_path = os.path.dirname(os.path.abspath(__file__))
repo_path = '/'.join(repo_path.split('/')[:-1])

#load relevant variables from YAML
with open(f'{repo_path}/data/user_data.yml', 'r') as f:
    user_data = yaml.safe_load(f)

    gfold_sif = user_data['paths']['gfold_sif']
    mmseqs_sif = user_data['paths']['mmseqs_sif']

def run_gfold(q,init_file,final_file,out_file):
    '''
    Run GFold enrichment calculations on the data in the provided files.
    
    Parameters
    ----------
    q: float
        the quantile of the estimated LFC distribution to report
    init_file: str
        the file path for the data file containing count information to use as the denomenator of the FC
    final_file: str
        the file path for the data file containing count information to use as the numerator of the FC
    out_file: str
        the desired path for the GFold result to be saved to
        
    Returns
    -------
    None
    '''
    cmd = ['singularity','exec',gfold_sif,
           'gfold','diff',
           '-norm','Count',
           '-s1',init_file,
           '-s2',final_file,
           '-suf','.tsv',
           '-o',out_file,
           '-sc',str(q)]
    gfold_result = subprocess.run(cmd)
    assert gfold_result.returncode == 0
    
def aggregate_and_calculate_fc(data,agg_funcs,agg_col,pair_df,gfold_quantiles,pool,tmpdir='tmp'):
    '''
    Calculate enrichment and GFold values for the specified pairs of groups of columns.
    
    Parameters
    ----------
    data: DataFrame
        pandas DataFrame containing the read count, sequence, and clustering information
    agg_funcs: dict
        mapping of columns to functions to aggregate them with when aggregating within clusters
    agg_col: str
        name of the column containing cluster label info
    pair_df: DataFrame
        pandas DataFrame containing information on which groups of columns to compare
        must contain the following columns
            initial: the names of the read count columns to use as the denominator of the FC calculation (with names separated by '+")
            final: the names of the read count columns to use as the numerator of the FC calculation (with names separated by '+")
            name: the base name to give the resulting GFold and enrichment columns
    gfold_quantiles: list
        list of quantile values use for GFold calculations
        one column entry in this list will be created for each row in pair_df
    pool: Pool
        the parallel pool to use for GFold calculations
        
    Returns
    -------
    result: DataFrame
        pandas DataFrame with the same columns as data, plus gfold and enrichment info aggregated over the labels in agg_col
        each row corresponds to a unique value in agg_col
    '''
    #aggregate on requested column
    print(f'Aggregating on {agg_col}.',flush=True)
    result = data.groupby(by=agg_col,sort=False).aggregate(agg_funcs)
    result[agg_col] = result.index #move sequence back to a column
    result.index = range(len(result)) #reset index
    print(f'{len(result)} unique {agg_col} values.',flush=True)
    
    #run GFOLD on aggregated data
    os.mkdir(tmpdir)
    os.mkdir(f'{tmpdir}/inputs')
    os.mkdir(f'{tmpdir}/outputs')
    
    #create datasets to compare
    datasets = set(pair_df['initial'].tolist()+pair_df['final'].tolist())
    short = {}
    for ds_idx,ds in enumerate(datasets):
        count_cols = ds.split('+')
        # short_name = '+'.join([x.split('_')[0] for x in count_cols])
        short[ds] = ds_idx#short_name
        
        for_gfold = result[['sequence_id']+count_cols].copy()
        for_gfold['sum'] = for_gfold[count_cols].sum(axis=1)
        
        #add fake columns for GFOLD
        for_gfold[['2','4','5']] = np.nan
        
        #rearrange columns to match GFOLD:
        #GeneSymbol, GeneName, Read Count, Gene exon length, RPKM
        for_gfold = for_gfold[['sequence_id','2','sum','4','5']]
        for_gfold.to_csv(f'{tmpdir}/inputs/{ds_idx}_{agg_col}.tsv',sep='\t',index=False,header=False)
        
    #run GFOLD in parallel for each user-specified quantile
    pool.starmap(run_gfold, ((q,
                              f'{tmpdir}/inputs/{short[row["initial"]]}_{agg_col}',
                              f'{tmpdir}/inputs/{short[row["final"]]}_{agg_col}',
                              f'{tmpdir}/outputs/{row["name"]}_{agg_col}_{str(q).split(".")[1]}.tsv') for q in gfold_quantiles for i,row in pair_df.iterrows()),
                 chunksize=1)
    
    #collect GFOLD results
    for q in gfold_quantiles:
        q_str = str(q).split(".")[1]
        
        for name in pair_df['name']:
            gfold_name = f'gfold{q_str}_{name}'
            gfold = pd.read_csv(f'{tmpdir}/outputs/{name}_{agg_col}_{q_str}.tsv',sep='\t',header=9,
                                names=['sequence_id','GeneName',gfold_name,'E-FDR','log2fdc','1stRPKM','2ndRPKM'])
            gfold = gfold[['sequence_id',gfold_name]]

            result = result.merge(gfold,on='sequence_id',how='outer')
        print(f'{len(result)} unique {agg_col} values after GFOLD.',flush=True)
    
    #calculate raw log-fold-change values
    for i,row in pair_df.iterrows():
        init_cols = row['initial'].split('+')
        init_count = result[init_cols].sum(axis=1)
        init_frac = init_count/init_count.sum()
        
        final_cols = row['final'].split('+')
        final_count = result[final_cols].sum(axis=1)
        final_frac = final_count/final_count.sum()
        
        if (max(final_count.sum(),init_count.sum())) == 0:
            print(row['initial'], row['final'], flush=True)
        
        fudge_factor = 1/(max(final_count.sum(),init_count.sum())) #use this to avoid divide-by-zero errors
        
        result = result.assign(**{f'enrich_{row["name"]}': np.log2((final_frac+fudge_factor)/(init_frac+fudge_factor))})
    
    #clean up tmp folder    
    subprocess.run(['rm','-r', tmpdir])
    
    #remove unnecessary columns
    for col in ['X','duplicate_count','VHH_duplicate_count','dist_nearest','centroid','centroid_id','family_homology_CDR3','is_outlier']:
        if col in result.columns:
            del result[col]
    
    return result

def translate_VHH_end_scan(dna,start_aa,end_aa,start_cap,end_cap):
    '''
    Find the VHH-encoding region of a DNA sequence and translate it to animo acids.
    
    Parameters
    ----------
    dna: str
        the DNA sequene to be translated
    start_aa: str
        the expected N-terminal amino acid sequence
    end_aa: str
        the expected C-terminal amino acid sequence
    start_cap: int
        the maximum number of bases to scan into the 5' end of dna when looking for start_aa
    end_cap: int
        the maximum number of bases to scan into the 3' end of dna when looking for end_aa
        
    Returns
    -------
    dna: str
        the same DNA sequence that was given as input
    aa: str
        the translated amino acid sequence that has the closest match to start_aa and end_aa
    best_match_start: int
        the number of amino acids that match start_aa in the best translation
    best_i_start: int
        the position in the DNA sequence that the translation starts at
    best_f: int
        the frame of the best translation
    best_match_end: int
        the number of amino acids that match end_aa in the best translation
    best_i_end: int
        the position in the DNA sequence that the translation ends at
    '''
    start_aa = np.array(list(start_aa))
    start_len = len(start_aa)
    
    #check all frames
    best_match_start = 0
    best_aa = ''
    best_i_start = np.nan
    best_f = np.nan
    for f in range(3):
        
        L = (len(dna)-f) - (len(dna)-f)%3
        aa = np.array(Seq(dna[f:L+f]).translate())
        
        #slide start_aa along translated seq to find best match (stop after first 40aa)
        for i in range(1,min(start_cap,len(aa))):
            
            #take the last i residues of start_aa (or the whole thing if i>start_len)
            start_segment = start_aa[-min(i,start_len):]
            
            #take the first i residues of aa (or a start_len stretch inside aa)
            aa_segment = aa[max(0,i-start_len):i]
            
            assert len(start_segment) == len(aa_segment)
            
            #count mismatches + non-overlap
            match = np.sum(start_segment == aa_segment)
            
            if match > best_match_start:
                best_match_start = match
                #add on the first piece of start_aa if best overlap is not a full overlap
                best_aa = np.concatenate([start_aa[:max(start_len-i,0)],aa[max(i-start_len,0):]],axis=0)
                best_i_start = i
                best_f = f
    
    aa = best_aa
    aa_len = len(aa)
        
    end_aa = np.array(list(end_aa))
    end_len = len(end_aa)
    
    best_match_end = 0
    best_aa = ''
    best_i_end = np.nan
    #given forward translation, find end point by sliding end_aa along the translation backwards
    for i in range(1,min(end_cap,len(aa))):

        #take the first i residues of end_aa (or the whole thing if i>end_len)
        end_segment = end_aa[:min(i,end_len)]

        #take the last i residues of aa (or a end_len stretch inside aa)
        aa_segment = aa[-i:min(aa_len-i+end_len,aa_len)]

        assert len(end_segment) == len(aa_segment)

        #count mismatches + non-overlap
        match = np.sum(end_segment == aa_segment)

        if match > best_match_end:
            best_match_end = match
            #add on the last piece of end_aa if best overlap is not a full overlap
            best_aa = np.concatenate([aa[:min(aa_len-i+end_len,aa_len)],end_aa[min(i,end_len):]],axis=0)
            best_i_end = i
                    
    return dna, ''.join(best_aa), best_match_start, best_i_start, best_f, best_match_end, best_i_end

def translate_wrapper(args):
    '''
    wrapper for translate_VHH_end_scan to make parallelization easier
    '''
    return translate_VHH_end_scan(*args)
    
def get_anarci_alignment(seq):
    '''
    Get ANARCI annotations for a VHH sequence.
    
    Parameters
    ----------
    seq: str
        the animo acid sequence to be annotated
        
    Returns
    -------
    seq: str
        the same sequence that was given as input
    annot: tuple
        tuple of numpy arrays with complete annotation info
        annot[0] is a character array with the amino acid sequence
        annot[1] is an integer array such that annot[1][i] is th IMGT position of annot[0][i]
    CDR1: str
        the CDR1 amino acid sequence
    CDR2: str
        the CDR2 amino acid sequence
    CDR3: str
        the CDR3 amino acid sequence
    ANARCI_success: bool
        indicates if the ANARCI parsing was successful
    '''
    sequences = [('id',seq)]
    numbering, _, _ = anarci(sequences, scheme="imgt", output=False, allowed_species=['alpaca'])
    
    assert len(numbering)==1 #only one sequence result
    
    #fail if no parse can be determined
    if numbering[0] is None:
        return seq, (np.array([]),np.array([])), '*', '*', '*', False
    #fail if parse contains multiple domains
    if len(numbering[0]) != 1:
        return seq, (np.array([]),np.array([])), '*', '*', '*', False
        
    start_idx = numbering[0][0][1]
    end_idx = numbering[0][0][2]
    numbering = numbering[0][0][0]
        
    #fill in the numbering to get to 128aa (this is canonical VHH length for IMGT)
    numbering = numbering + [((i,' '),'-') for i in range(numbering[-1][0][0]+1,129)]
    
    #add back in any residues trimmed off at the start
    numbering = [((i-start_idx+1,' '),seq[i]) for i in range(start_idx)] + numbering
    
    #add back in any residues trimmed off at the end
    numbering = numbering + [((128+i-end_idx,' '),seq[i]) for i in range(end_idx+1,len(seq))]
    
    seq_array = np.array([pos[1] for pos in numbering])
    num_array = np.array([pos[0][0] for pos in numbering])
    
    return (seq, (seq_array, num_array),
            ''.join(seq_array[(num_array>=27)&(num_array<=38)]).replace('-',''), #CDR1
            ''.join(seq_array[(num_array>=56)&(num_array<=65)]).replace('-',''), #CDR2
            ''.join(seq_array[(num_array>=105)&(num_array<=117)]).replace('-',''),#CDR3
            True)
            
def weighted_anarci_dist(CDR_list1,CDR_list2,weight_scheme,max_len_diffs):
    '''
    Calculates the weighted CDR distance (in the range [0,1]) between a pair of sequences.
    Each mismatch is weighted according to which CDR it's in.
    Pairs of sequences with CDRs that are too different in length are assigned a distance of 1.
    
    Parameters
    ----------
    CDR_list1: list
        list containing the CDR sequences for the first sequence
        CDR_list1[i] is the CDRi sequence
    CDR_list2: list
        list containing the CDR sequences for the second sequence
        CDR_list2[i] is the CDRi sequence
    weight_scheme: list
        list of weights for mismatches in each CDR
        weight of mismatch for CDRi is weight_scheme[i]
    max_len_diffs: list
        list of maximum allowed differene in length for each CDR
        max different for CDRi is max_len_diffs[i]
        
    Returns
    -------
    dist: float
        value in the range [0,1] describing the difference between the pair of sequences
    '''
    #set dist to 100% if either of the pair had no ANARCI parse
    if (CDR_list1[0] == '*') or (CDR_list2[0] == '*'):
        return 1
        
    #set dist to 100% if any CDR pair is too different in length
    for i in range(len(CDR_list1)-1,-1,-1): #do this in reverse order because most likely to have CDR3 diff so will return faster
        if abs(len(CDR_list1[i]) - len(CDR_list2[i])) > max_len_diffs[i]:
            return 1
    
    dist_tot = 0
    weight_tot = 0
    for i in range(len(CDR_list1)):
        dist = distance(CDR_list1[i],CDR_list2[i]) #edit distance between CDR pair
        dist_tot += dist*weight_scheme[i] #add weighted dist to total
        weight_tot += np.max([len(CDR_list1[i]),len(CDR_list2[i])])*weight_scheme[i] #add weight to total
        
    return dist_tot/weight_tot #compute weighted percent identity
    
def weighted_anarci_dist_no_align(CDR_list1, CDR_list2, weight_scheme):
    
    #set dist to 100% if either of the pair had no ANARCI parse
    if np.any(CDR_list1[0] == '*') or np.any(CDR_list2[0] == '*'):
        return 1
        
    dist_tot = 0
    weight_tot = 0
    for i in range(3):
        dist_tot += np.sum(CDR_list1[i]!=CDR_list2[i])*weight_scheme[i]
        weight_tot += len(CDR_list1[i])*weight_scheme[i]
        
    return dist_tot/weight_tot
    
def ANARCI_dist_row(CDR_list1,CDR_list_list,weight_scheme,max_len_diffs):
    '''
    Wrapper to calculate distances from a single VHH sequence to all VHHs in a list
    Intended for calculation ofone row in a pairwise distance matix.
    
    Parameters
    ----------
    CDR_list1: list
        list containing the CDR sequences for the first sequence
        CDR_list1[i] is the CDRi sequence
    CDR_list_list: list
        list containing the lists of CDR sequences for all sequences to compare to
        CDR_list_list[j][i] is the CDRi sequence for VHH j
    weight_scheme: list
        list of weights for mismatches in each CDR
        weight of mismatch for CDRi is weight_scheme[i]
    max_len_diffs: list
        list of maximum allowed differene in length for each CDR
        max different for CDRi is max_len_diffs[i]
        
    Returns
    -------
    dists: list
        list of distances between each VHH in CDR_list_list and CDR_list1
    '''
    return [weighted_anarci_dist(CDR_list1,CDR_list2,weight_scheme,max_len_diffs) for CDR_list2 in CDR_list_list]
    
def ANARCI_dist_pairs(pair_list,weight_scheme,max_len_diffs):
    '''
    Wrapper to calculate distances berween arbitrary pairs of VHHs

    Parameters
    ----------
    pair_list: list
        list containing the CDR sequences for the pairs to be compared
        pair_list[j][0][i] is the CDRi sequence for the first member of pair j
        pair_list[j][1][i] is the CDRi sequence for the second member of pair j
    weight_scheme: list
        list of weights for mismatches in each CDR
        weight of mismatch for CDRi is weight_scheme[i]
    max_len_diffs: list
        list of maximum allowed differene in length for each CDR
        max different for CDRi is max_len_diffs[i]
        
    Returns
    -------
    dists: list
        list of distances for each VHH pair in pair_list
    '''
    return [weighted_anarci_dist(*pair,weight_scheme,max_len_diffs) for pair in pair_list]
    
def fill_in_weighted_anarci_dists(dist_df, info_df, dist_col, weight_scheme, max_len_diff, pool, ncpus):
    '''
    Given a DataFrame of VHH pairs with some distances calculated and a set of pairs to get distances between,
    calculate any distances that have not been already and add them ot the distance DataFrame.
    
    Parameters
    ----------
    dist_df: DataFrame
        the DataFrame that has some pre-calculated distances in it
        must contain a column name matching the dist_col parameter, as well as the following columns:
            query: the name of the "query" sequence in the pair
            target: the name of the "target" sequence in the pair
    info_df: DataFrame
        the DataFrame that has pairs of sequences that need distances
        must be indexed by the values in dist_df['query'] and dist_df['target']
        must containt the following columns:
            CDR1: the CDR1 amino acid sequence
            CDR2: the CDR2 amino acid sequence
            CDR3: the CDR3 amino acid sequence
    dist_col: str
        the column name in dist_df that contains the calculated distances, or NaN if they have not been calculated yet
    weight_scheme: list
        list of weights for mismatches in each CDR to use in distance calculations
        weight of mismatch for CDRi is weight_scheme[i]
    max_len_diffs: list
        list of maximum allowed differene in length for each CDR to use in distance calculations
        max different for CDRi is max_len_diffs[i]
    pool: Pool
        a parallel pool object to use for parallel processing of distance calculations
    ncpus: int
        the number of cpus that pool has access to
        
    Returns
    -------
    dist_df: DataFrame
        the updated version of dist_df with new distances filled in
    missing_idx: Series
        pandas Series of bools indicating if the corresponding row in dist_df was filled in
    '''
    missing_idx = dist_df[dist_col].isna()
    N_fill = missing_idx.sum()
    if N_fill:
        
        batch_size = max(N_fill//(ncpus*4),100)
        print(f'Filling in {missing_idx.sum()}/{len(dist_df)} missing dists in batches of {batch_size}.',flush=True)
        #add final small batch if needed
        split_points = np.arange(0,N_fill,batch_size)
        if split_points[-1] < N_fill:
            split_points = np.concatenate((split_points,[N_fill]))
        
        #calculate distances
        dist_sets = pool.map(partial(ANARCI_dist_pairs,
                                     weight_scheme=weight_scheme,
                                     max_len_diffs=max_len_diff),([[info_df.loc[row['query'],['CDR1','CDR2','CDR3']].tolist(),
                                                                    info_df.loc[row['target'],['CDR1','CDR2','CDR3']].tolist()] for _,row in dist_df[missing_idx].iloc[s:split_points[i+1],:].iterrows()] for i,s in enumerate(split_points[:-1])))
        dist_df.loc[missing_idx,dist_col] = np.concatenate(dist_sets,axis=0)
        
    return dist_df,missing_idx
    
def weighted_anarci_dist_all_pairs(CDR_list1, CDR_list2, same, weight_scheme, max_len_diffs):
    '''
    Calculate pairwise distances between all pairs of sequences across the two lists provided.
    
    Parameters
    ----------
    CDR_list1: list
        list of lists of strs
        the inner lists each contain the CDR amino acid sequences for a single VHH in set 1
    CDR_list2: list
        list of lists of strs
        the inner lists each contain the CDR amino acid sequences for a single VHH in set 2
    same: bool
        indicates if set 1 and set 2 contain the exact same sequences
        if True, only upper-diagonal pairwise distances are calculated
    weight_scheme: list
        list of weights for mismatches in each CDR to use in distance calculations
        weight of mismatch for CDRi is weight_scheme[i]
    max_len_diffs: list
        list of maximum allowed differene in length for each CDR to use in distance calculations
        max different for CDRi is max_len_diffs[i]
    pool: Pool
        a parallel pool object to use for parallel processing of distance calculations
    
    Returns
    -------
    dists: array
        numpy array containing the pairwise disances
        dists[i,j] is the distance between CDR_list1[i] and CDR_list2[j]
        diagonal and below entries are 0 if same is True
    '''
    #if given two lists, compare all cross-list pairs
    if not same:
        dists = np.zeros([len(CDR_list1),len(CDR_list2)])*np.nan
        for i,cdrs1 in enumerate(CDR_list1):
            for ii,cdrs2 in enumerate(CDR_list2):
                dists[i,ii] = weighted_anarci_dist(cdrs1,cdrs2,weight_scheme,max_len_diffs)
    #otherwise, compare all pairs within the single list
    else:
        dists = np.zeros([len(CDR_list1)]*2)*np.nan
        for i,cdrs1 in enumerate(CDR_list1[:-1]):
            for ii,cdrs2 in enumerate(CDR_list1[i+1:]):
                dists[i,ii+i+1] = weighted_anarci_dist(cdrs1,cdrs2,weight_scheme,max_len_diffs)
        dists[np.diag_indices(dists.shape[0])] = 0
                
    return dists
    
def batched_pairwise_weighted_anarci_dist(CDR_list, batch_size, weight_scheme, max_len_diffs, pool):
    '''
    Split pairwise distance calculations for all pairs within a given set into batches and calculate each batch in parallel.
    
    Parameters
    ----------
    CDR_list: list
        list of lists of strs
        the inner lists each contain the CDR amino acid sequences for a single VHH
        pairwise distancess between all VHHs in CDR_list will be calculated
    batch_size: int
        the number of VHHs to put in each batch (resulting matric will have batch_size**2 distances)
    weight_scheme: list
        list of weights for mismatches in each CDR to use in distance calculations
        weight of mismatch for CDRi is weight_scheme[i]
    max_len_diffs: list
        list of maximum allowed differene in length for each CDR to use in distance calculations
        max different for CDRi is max_len_diffs[i]
        
    Returns
    -------
    dists: array
        numpy array containing the pairwise disances
        dists[i,j] is the distance between CDR_list[i] and CDR_list[j]
    '''
    N_seqs = len(CDR_list)
    
    N_batches = int(np.ceil(N_seqs/batch_size))
    batches = ([CDR_list[b1*batch_size:(b1+1)*batch_size], CDR_list[b2*batch_size:(b2+1)*batch_size], b1==b2] for b1 in range(N_batches) for b2 in range(b1,N_batches))
    dist_batches = pool.starmap(partial(weighted_anarci_dist_all_pairs,
                                        weight_scheme=weight_scheme,
                                        max_len_diffs=max_len_diffs), batches)
    dists = np.zeros([N_seqs]*2)*np.nan
    i = 0
    for b1 in range(N_batches):
        for b2 in range(b1,N_batches):
            dists[b1*batch_size:(b1+1)*batch_size,b2*batch_size:(b2+1)*batch_size] = dist_batches[i]
            i+=1
    idxs = np.tril_indices(dists.shape[0], k=-1)
    dists[idxs] = dists.T[idxs]
    return dists
    
def weighted_anarci_dist_all_pairs_no_align(CDR_list1, CDR_list2, same, weight_scheme):
    #if given two lists, compare all cross-list pairs
    if not same:
        dists = np.zeros([len(CDR_list1),len(CDR_list2)])*np.nan
        for i,cdrs1 in enumerate(CDR_list1):
            for ii,cdrs2 in enumerate(CDR_list2):
                dists[i,ii] = weighted_anarci_dist_no_align(cdrs1,cdrs2,weight_scheme)
    #otherwise, compare all pairs within the single list
    else:
        dists = np.zeros([len(CDR_list1)]*2)*np.nan
        for i,cdrs1 in enumerate(CDR_list1[:-1]):
            for ii,cdrs2 in enumerate(CDR_list1[i+1:]):
                dists[i,ii+i+1] = weighted_anarci_dist_no_align(cdrs1,cdrs2,weight_scheme)
        dists[np.diag_indices(dists.shape[0])] = 0
                
    return dists
    
def batched_pairwise_weighted_anarci_dist_no_align(CDR_list, batch_size, weight_scheme, pool):
    N_seqs = len(CDR_list)
    
    N_batches = int(np.ceil(N_seqs/batch_size))
    batches = ([CDR_list[b1*batch_size:(b1+1)*batch_size], CDR_list[b2*batch_size:(b2+1)*batch_size], b1==b2] for b1 in range(N_batches) for b2 in range(b1,N_batches))
    dist_batches = pool.starmap(partial(weighted_anarci_dist_all_pairs_no_align,
                                        weight_scheme=weight_scheme), batches)
    dists = np.zeros([N_seqs]*2)*np.nan
    i = 0
    for b1 in range(N_batches):
        for b2 in range(b1,N_batches):
            dists[b1*batch_size:(b1+1)*batch_size,b2*batch_size:(b2+1)*batch_size] = dist_batches[i]
            i+=1
    idxs = np.tril_indices(dists.shape[0], k=-1)
    dists[idxs] = dists.T[idxs]
    return dists
    
def weighted_anarci_dist_all_pairs_filtered(input_tuple):
    CDR_list1, CDR_list2, start1, start2, max_dist, cdr3_weight, max_len_diff = input_tuple
    
    weight_scheme = [1,1,cdr3_weight]
    max_len_diffs = [max_len_diff]*3
    
    dists = []
    
    #if given two lists, compare all cross-list pairs
    if start1!=start2:
        for i,cdrs1 in enumerate(CDR_list1):
            for ii,cdrs2 in enumerate(CDR_list2):
                d = weighted_anarci_dist(cdrs1,cdrs2,weight_scheme,max_len_diffs)
                if d <= max_dist:
                    dists.append((i+start1, ii+start2, {'dist':d}))
    #otherwise, compare all pairs within the single list
    else:
        for i,cdrs1 in enumerate(CDR_list1[:-1]):
            for ii in range(i+1,len(CDR_list1)):
                cdrs2 = CDR_list1[ii]
                d = weighted_anarci_dist(cdrs1,cdrs2,weight_scheme,max_len_diffs)
                if d <= max_dist:
                    dists.append((i+start1, ii+start2, {'dist':d}))
    return dists
    
def get_pairwise_cluster_rep_distances(df,id_cols,n_reps):
    '''
    For a given list of cluster label columns, calculate pairwise "rep distances" between the columns.
    rep_distance is defined as the number of representatives from either clustering that are not representatives in both clusterings.
    A sequence is a representative in a clustering if it is one of the most abundant sequence in its cluster.
    
    Parameters
    ----------
    df: DataFrame
        the DataFrame with the clustering info
        must contain the columns listed in id_cols and a column named sequence_id that contains a unique label for each sequence
    id_cols: list
        list of column names in df that contain cluster labels to be compared
    n_reps: int
        how many representatives to take from each cluster
        the most abundant members are used as representatives
        
    Returns
    -------
    clust_dists: array
        numpy array where clust_dists[i,j] is the rep_distance between id_cols[i] and id_cols[j]
    '''
    df = df.sort_values(by='VHH_duplicate_count',ascending=False)
    
    #cluster the clusterings by top N overlap
    #get dists
    clust_dists = np.zeros([len(id_cols)]*2)*np.nan
    for i,col1 in enumerate(id_cols[:-1]):
        reps1 = df.groupby(col1).head(n_reps)
        for ii,col2 in enumerate(id_cols[i+1:]):
            reps2 = df.groupby(col2).head(n_reps)
            new_seqs = len(set(reps1['sequence_id']).symmetric_difference(set(reps2['sequence_id'])))
            clust_dists[i,ii+i+1] = new_seqs
            clust_dists[ii+i+1,i] = new_seqs
    clust_dists[range(len(id_cols)),range(len(id_cols))] = 0
    
    return clust_dists

def agg_clust_to_max_extra_reps(df,id_cols,dists,method,n_reps,max_extra_reps):
    '''
    Recursively find groups of clusterings that have similar enough sets of representatives that it makes sense to recycle distance calculations between them.
    For use in silhouette scoring to improve compute time.
    
    Parameters
    ----------
    df: DataFrame
        the DataFrame with the clustering info
        must contain the columns listed in id_cols and a column named sequence_id that contains a unique label for each sequence
    id_cols: list
        list of column names in df that contain cluster labels to be grouped
    dists: array
        numpy array where clust_dists[i,j] is the rep_distance between id_cols[i] and id_cols[j]
    method: str
        name of agglomerative clustering method to use when clustering columns
    n_reps: int
        how many representatives to take from each cluster of sequences
        the most abundant members are used as representatives
    max_extra_reps: int
        the maximum number representatives allowed in a cluster after adding in representatives of clusters in other clusterings
        
    Returns
    -------
    labels: array
        numpy array where new_labels[i] is the group that id_cols[i] belongs to
    '''
    #find how many extra reps are necessary with this group of columns
    _, max_reps = get_fam_reps(id_cols,df,n_reps)
    print(f'{len(id_cols)} columns with max_reps={max_reps}',flush=True)
    
    # if too many extra reps are needed, recursively split into smaller groups
    if max_reps > max_extra_reps:
        #cluster into 2 groups
        clusterer = AgglomerativeClustering(n_clusters=2,metric='precomputed',linkage=method)
        labels = clusterer.fit_predict(dists)
        
        print(f'\t split into {np.sum(labels==0)} and {np.sum(labels==1)} columns',flush=True)
        
        #send each subgroup into recursive clustering
        new_labels = np.zeros(labels.shape)
        for l in range(2):
            label_match = labels==l
            col_subset = [col for i,col in enumerate(id_cols) if label_match[i]]
            dist_subset = dists[:,label_match][label_match,:]
            split_labels = agg_clust_to_max_extra_reps(df,col_subset,dist_subset,method,n_reps,max_extra_reps)
            new_labels[label_match] = split_labels + np.max(new_labels) + 1
            
        return new_labels
    
    #otherwise keep labels as-is
    else:
        return np.zeros(len(id_cols))

def get_fam_reps(id_cols,df,n_reps):
    '''
    Get the set of all representatives for a group of cluster label columns.
    
    Parameters
    ----------
    id_cols: list
        list of column names in df that contain cluster labels to be grouped
    df: DataFrame
        the DataFrame with the clustering info
        must contain the columns listed in id_cols and a column named sequence_id that contains a unique label for each sequence
    n_reps: int
        how many representatives to take from each cluster of sequences
        the most abundant members are used as representatives
        
    Returns
    -------
    all_reps: DataFrame
        contains the same columns as df, but only the rows corresponding to representatives of the clusterings in id_cols
    max_reps: int
        the size of the largest cluster in any clustering in id_cols, after excluding sequences that are not representatives of any columns
    '''
    #collect set of all possible family representatives
    all_reps = pd.DataFrame()
    for id_col in id_cols:
        id_reps = df.groupby(id_col).head(n_reps)
        all_reps = pd.concat([all_reps,id_reps],axis=0).drop_duplicates('VHH')
    all_reps = all_reps.set_index('sequence_id')

    #figure out largest number of reps for a cluster in any clustering
    max_reps = 0
    for id_col in id_cols:
        max_reps = max(max_reps,all_reps.groupby(id_col).size().max())

    return all_reps, max_reps
    
def approx_avg_internal_anarci_dist(df,n_sample,cdr_weights,max_len_diff,pool):
    '''
    Approximate the average pairwise distance between VHHs in the given DataFrame.
    
    df: DataFrame
        pandas DataFrame containing info about the sequences to be compared
        must have the following columns:
            CDR1: the CDR1 amino acid sequence
            CDR2: the CDR2 amino acid sequence
            CDR3: the CDR3 amino acid sequence
    n_sample: int
        number of pairs to sample when approximating the average
        if the number of pairs within df is less than n_sample, all pairwise distances are calculated
    cdr_weights: list
        list of weights for mismatches in each CDR to use in distance calculations
        weight of mismatch for CDRi is cdr_weights[i]
    max_len_diffs: list
        list of maximum allowed differene in length for each CDR to use in distance calculations
        max different for CDRi is max_len_diffs[i]
    pool: Pool
        a parallel pool object to use for parallel processing of distance calculations
        
    Returns
    -------
    avg_dist: float
        the approximated average distance
    '''
    N = len(df)
    #if there is only one item, it is 0 from itself
    if N == 1:
        dists = 0
    #if the dataframe is small enough, no sampling needed
    elif N*(N-1)/2 <= n_sample:
        dists = pool.starmap(weighted_anarci_dist,((row1[['CDR1','CDR2','CDR3']].tolist(),
                                                    row2[['CDR1','CDR2','CDR3']].tolist(),
                                                    cdr_weights,
                                                    max_len_diff) for i,(_,row1) in enumerate(df.iloc[-1:,:].iterrows()) for _,row2 in df.iloc[i+1:,:].iterrows()))
    else:
        #generate random samples
        cdr_pairs = []
        for i in range(n_sample):
            pair = df.sample(2,replace=False,axis=0)
            cdr_pairs.append([pair.iloc[0,:][['CDR1','CDR2','CDR3']].tolist(),
                              pair.iloc[1,:][['CDR1','CDR2','CDR3']].tolist()])
        dists = pool.starmap(weighted_anarci_dist,((pair[0],
                                                    pair[1],
                                                    cdr_weights,
                                                    max_len_diff) for pair in cdr_pairs))
    return np.mean(dists)
    
def run_mmseqs_clust(seq_df, min_id, method, tmp_dir, ncpus):
    '''
    Wrapper for running MMseqs2 clustering
    
    Parameters
    ----------
    seq_df: DataFrame
        pandas DataFrame containing sequence info for the sequences to be clustered
        must contain the following columns
            VHH: the amino acid sequence
            sequence_id: unique identifier for each sequence
    min_id: float
        minimum identity (in the range [0,1]) for MMseqs to group sequences together
    method: str
        the name of the clustering method for MMseqs to use (setcover or conncomp)
    tmp_dir: str
        the directory to put temp files in
    ncpus: int
        the number of cores to parallelize MMseqs calculations over
        
    Returns
    -------
    seq_df: DataFrame
        updated version of seq_df with the MMseqs cluster labels listed in the mmseqs_id column
    '''
    if method == 'setcover':
        method_index = '0'
    elif method == 'conncomp':
        method_index = '1'
    else:
        raise ValueError('Invalid mmseqs clustering method.')
    
    #write fasta file
    with open(f'{tmp_dir}/mmseqs_in.fasta','w') as f:
        for i,row in seq_df.iterrows():
            f.write(f">{row['sequence_id']}\n{row['VHH']}\n") #Index here is the intermediate_id
            
    #run mmseqs clustering
    result = subprocess.run(['singularity', 'exec', mmseqs_sif, 'mmseqs', 'easy-cluster',
                            f'{tmp_dir}/mmseqs_in.fasta', f'{tmp_dir}/mmseqs_out', f'{tmp_dir}/tmp',
                             '-s', '7.5', #sensitivity set to max
                             '-c', str(max(0,(min_id - 0.2))), #set this lower than seq-id cutoff so it doesn't have an impact
                             '--min-seq-id', str(min_id), #only cluster seqs more similar than user input
                             '-e', 'inf', #max e-value to be clustered (set to keep all)
                             '-v', '1', #verbosity
                             '--cluster-steps', '3', #use faster cascaded clustering method
                             '--seq-id-mode', '0', #use alignment length to normalize
                             '--cluster-mode', method_index, #use user-specified method
                             '--alignment-mode', '3', #get percent identity, rather than alignment score
                             '--max-seqs', '1000', #get top 1000 matches to each query
                             '--gap-open', '20', #set high gap-opening penalty to prevent excessive gaps
                             '--gap-extend', '3',#set high gap-extend penalty to prevent excessive gaps
                             '--similarity-type', '2', #use sequence identity, not score
                             '--threads', str(ncpus)]) #how many cpus to use
    assert result.returncode == 0
    
    #read in clusters
    mmseqs_clusters = pd.read_csv(f'{tmp_dir}/mmseqs_out_cluster.tsv',names=['mmseqs_id','sequence_id'],sep='\t')
    
    #remove old clusters and add new ones
    seq_df = seq_df.drop(columns=['mmseqs_id'],errors='ignore')
    seq_df = seq_df.merge(mmseqs_clusters,on='sequence_id',how='left')
    
    #clean up
    for f in glob(f'{tmp_dir}/*'):
        subprocess.run(['rm', '-r', f])
        
    return seq_df

def collect_seqs(files):
    '''
    Concatenate data from all tsv files in the provided list.
    No deduplication is performed.
    
    Parameters
    ----------
    files: list
        the list of file names to concatenate
        these should be the *_db-pass.tsv files generated by assemble_and_filter_reads.py
        
    Returns
    -------
    data: DataFrame
        pandas DataFrame with all requested files concatenated.
    '''
        
    print('Reading the following files:')
    print('\n'.join([f.split('/')[-1] for f in files]))
    data = pd.concat([pd.read_csv(tsv,sep='\t') for tsv in tqdm(files)],axis=0)
    #drop less useful info
    data = data[["sequence", "locus", "junction", "junction_aa", "v_call", "d_call", "j_call", "duplicate_count"]]
    
    print(f'{len(data)} rows in global set after loading tsvs.', flush=True)
    return data

def annotate_and_filter_seqs(args, data, pool, ncpus, agg_funcs):
    '''
    Take an input DataFrame, remove bad sequeces from it, and use ANARCI to determine CDR sequences.
    Removes the following types of sequences:
        Sequences with internal stop codons
        Sequences with N or C termini that do not resemble VHHs
        Rare sequences
    Adds the following info columns for each sequence
        sequence_id: a unique string identifying each sequence
        VHH: the translated sequence
        start_match: the number of amino acids at the beginning of the translation that match args.start_seq
        end_match: the number of amino acids at the end of the translation that match args.end_seq
        start_i: the position in the DNA sequence where the translation started
        end_i: the position in the DNA sequence where the translation ended
        start_f: which frame the rtanslation starts in
        CDR1: the CDR1 amino acid sequence
        CDR2: the CDR2 amino acid sequence
        CDR3: the CDR3 amino acid sequence
        anarci_parse: a tuple of numpy arrays with annotations from ANARCI
            anarci_parse[0] is a character array with the amino acid sequence
            anarci_parse[1] is an integer array such that anarci_parse[1][i] is th IMGT position of anarci_parse[0][i]
        ANARCI_success: bool indicating if the ANARCI parsing was successful
    
    Parameters
    ----------
    args: parsed arguments
        The arguments passed as input into the main script - used for start and end point determination and filtering
        Must contain the following fields:
            start_seq: the amino acid sequence expected at the N-term of all VHHs
            end_seq: the amino acid sequence expected at the C-term of all VHHs
            max_start_scan: the maximal number of bases to scan into the 5' end of the DNA sequence when looking for a translation match to start_seq
            max_end_scan: the maximal number of bases to scan into the 3' end of the DNA sequence when looking for a translation match to end_seq
            min_start_match: the minimum number of amino acids matching start_seq for the translation to be considered successful
            min_end_match: the minimum number of amino acids matching end_seq for the translation to be considered successful
            min_count: the minimum number of read counts associated with an amino acid sequence for it to be retained in the dataset
    
    data: DataFramw
        pandas DataFrame containing the sequences to be annotated and filtered
        Must have the following columns:
            sequence: the DNA sequence of the VHH
            duplicate_count: the number of read counts associated with each DNA sequence
            
    pool: Pool
        a parallel pool object to use for parallel processing of translation and ANARCI annotation
        
    ncpus: int
        the number of cpus that pool has access to
        
    agg_funcs: dict
        dictionary mapping columns in data to functions that will be used to aggregate the data in those columns when aggregating by amino acid sequence
        
    Returns
    -------
    data: DataFrame
        The modified version of data including new annotation columns, and with the bad sequences filtered out
        
    vhh: DataFrame
        The same information as in data, but aggregated by amino acid sequence instead of DNA sequence
    '''
    # Aggregate duplicate DNA sequences
    data = data.groupby('sequence', sort=False).aggregate(func=agg_funcs).reset_index(drop=True)
    print(f'{len(data)} unique DNA sequences.', flush=True)

    # Parallel translation of sequences
    # Dynamically choose a good chunksize
    chunksize = max(len(data) // (ncpus * 4), 1000)
    
    # Efficient parallel translation
    results = pool.imap_unordered(
        translate_wrapper,
        zip(
            data['sequence'],
            repeat(args.start_seq),
            repeat(args.end_seq),
            repeat(args.max_start_scan),
            repeat(args.max_end_scan)
        ),
        chunksize=chunksize
    )
    
    # Collect results into DataFrame
    seq_info = pd.DataFrame(list(results), columns=['sequence','VHH','start_match','start_i','start_f','end_match','end_i'])

    # Merge translation results into original dataframe
    data = data.merge(seq_info, on='sequence', how='inner')

    # Filter out sequences with stop codons, poor start/end matches, or invalid translations
    data = data[
        data['VHH'].notna() &
        (~data['VHH'].str.contains(r'\*', regex=True)) &
        (data['start_match'] >= args.min_start_match) &
        (data['end_match'] >= args.min_end_match)
    ]
    print(f'{len(data)} sequences passed translation and motif filtering.', flush=True)

    # Compute VHH-level duplicate counts and merge
    vhh_counts = data.groupby('VHH',sort=False)['duplicate_count'].sum().rename('VHH_duplicate_count').reset_index()
    data = data.merge(vhh_counts, on='VHH')

    # Drop singleton VHHs
    data = data[data['VHH_duplicate_count'] >= args.min_count]
    print(f'{len(data)} sequences, encoding {data["VHH"].nunique()} unique VHHs after removing rare amino acid sequences (<{args.min_count}).', flush=True)

    # Reset index and assign sequence IDs
    data = data.reset_index(drop=True)
    data['sequence_id'] = data.index.map(lambda i: 'GS-' + str(i))

    # Drop duplicate full-length VHHs (keep most abundant DNA encoding)
    data = data.sort_values(by='duplicate_count', ascending=False)
    vhh = data.drop_duplicates('VHH', keep='first')

    print(f'{len(vhh)} unique VHHs after deduplication.', flush=True)

    # Get ANARCI annotations
    print(f'Getting ANARCI parses.', flush=True)
    anarci_info = pd.DataFrame(pool.map(get_anarci_alignment, vhh['VHH'], chunksize=1000),
                               columns=['VHH', 'anarci_parse', 'CDR1', 'CDR2', 'CDR3', 'ANARCI_success'])
    vhh = vhh.merge(anarci_info, on='VHH', how='left')
    print(f'{len(vhh)} rows after merging with ANARCI data.', flush=True)

    return data, vhh
        
def add_header(df, header_names, sample_df):
    '''
    Add a header to the given DataFrame with information about each column.
    
    Parameters
    ----------
    df: DataFrame
        pandas DataFrame to add the header to
        must have columns starting with 'gfold' or 'enrich' to add header info to
    header_names: list
        list of column names in sample_df to pull info from for header
    sample_df: DataFrame
        pandas DataFrame with metadata on columns in df
        must have columns matching header names and a column names no_rep with entries corresponding to suffixes of gfold and enrich columns in df
        
    Returns
    -------
    df: DataFrame
        an updated version of df containing the requested header
    '''
    header = []
    for col in df.columns:
        #for enrighment and gfold columns, look for label inside raw data column names
        if col.startswith('enrich') or col.startswith('gfold'):
            name = col.split('_')[1]
            matches = sample_df[sample_df['no_rep']==name]
        #for others check for exact match to find raw data columns
        else:
            matches = sample_df[sample_df['name']==col]
            
        if len(matches): #this means this is a column we want to label because it's has quantitative data
            
            assert len(matches['no_rep'].unique()) == 1 #make sure we're getting a single sample's worth of data
            
            match_names = matches[header_names].iloc[0,:]
            match_names[match_names.isna()] = ''
            header.append(tuple(match_names.tolist()+[col]))
            
        else: #fill in blank info for columns with sequence descriptors
            header.append(tuple(['']*len(header_names)+[col]))
            
    df.columns = pd.MultiIndex.from_tuples(header,names=header_names+['full_name'])
    return df
    
def pairwise_batch_generator(items,batch_size,extras):
    '''
    Generator function yielding batches to pass to a function that runs pairwise calculations.
    
    Parameters
    ----------
    items: iterable
        the items to be used in pairwise calculations
    batch_size: int
        the number of items to include on each side of the pairwise calculations in each batch
        the number of pairwise calculations will be batch_size**2
    extras: list
        additional positional arguments to yield with each batch
        
    Returns
    -------
    batch: tuple
        the batch to run pairwise calculations on
        batch[0] and batch[1] are the items for which pairwise calculations should be made
        batch[2] and batch[3] are the indexes in items corresponding to the first element in batch[0] and batch[1], respectively
        batch[4:] are the elements of extras
    '''
    N_batches = int(np.ceil(len(items)/batch_size))
    for b1 in range(N_batches):
        start1 = b1*batch_size
        end1 = (b1+1)*batch_size
        for b2 in range(b1,N_batches):
            start2 = b2*batch_size
            end2 = (b2+1)*batch_size
            
            yield items[start1:end1], items[start2:end2], start1, start2, *extras
    
def get_CDR_len_combo(row):
    
    cdr_id = 0
    for c in range(1,4):
        cdr_id += (1000**(3-c))*len(row[f'CDR{c}'])
        
    return cdr_id