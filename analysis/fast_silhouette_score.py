import numpy as np
import pandas as pd
import sys
import os
import subprocess
from sklearn.cluster import AgglomerativeClustering
from multiprocessing import Pool
from tqdm import tqdm
import argparse
from scipy.spatial.distance import squareform
from functools import partial

#get main dir for repo
repo_path = os.path.dirname(os.path.abspath(__file__))
repo_path = '/'.join(repo_path.split('/')[:-1])
sys.path.append(repo_path)
from utils import weighted_anarci_dist, get_anarci_alignment

def get_args():
    #get user inputs
    parser = argparse.ArgumentParser(prog='Sequence-based evaluation of clutering quality',
                                     description='Calculate the "fast silhouette" score based on CDR distance for a given (set of) clustering(s).')
    #I/O args
    parser.add_argument('--in_file',type=str,help='csv file containing sequence and cluster label info')
    parser.add_argument('--out_file',type=str,help='Name for the output file (can include file path).')
    parser.add_argument('--label_columns',type=str,nargs='*',help='Which column(s) contain the cluster labels to be evaluated')
    parser.add_argument('--drop_duplicates',type=str,default='VHH',help='Which column to use to drop duplicate entries before scoring. Set to "none" to skip this step.')
    
    
    #distance function args
    parser.add_argument('--CDR3_weight',type=float,default=3, help='Weight for CDR3 in weighted distance calculations. Uses a weight of 1 for CDR1 and CDR2.')
    parser.add_argument('--max_len_diff',type=int,default=3, help='Maximum allowed difference in CDR length. Distance is set to 1 if this is exceeded.')
    
    #scoring args
    parser.add_argument('--sample_N_pairs',type=int,default=10000,help='When calculating intra-family and inter-family distances, calcualte weighted distances for a random sampling of sample_N_pairs pairs of sequences.')
    
    return parser.parse_args()

def weighted_ANARCI_silhouette_fast(full_df, id_cols, dist_param_map, sample_N_pairs, pool, ncpus):
    '''
    Calculate the "fast Silhouette" score, defined as the average between-cluster distance minus the average within-cluster distance, approximated by randomly sampling within- and between-cluster pairs.
    
    Parameters
    ----------
    full_df: DataFrame
        pandas DataFrame with the sequence and clustering information
        must include columns matching those in id_cols as well as:
            CDR1: the CDR1 amino acid sequence
            CDR2: the CDR2 amino acid sequence
            CDR3: the CDR3 amino acid sequence
    id_cols: list
        list of column names in full_df with cluster labels to score
        each listed column will be scored separately
    dist_param_map: dict
        dictionary mapping column names in id_cols to distance function parameters
        dist_param_map[col][0] is the weight for CDR3 mismatches (with CDR1 and 2 having weight of 1)
        dist_param_map[col][1] is maximum allowed difference in length for any CDR
    sample_N_pairs: int
        how many within- and between-cluster pairs to sample when approximating the average distances
    pool: Pool
        a parallel pool object to use for parallel processing of distance calculations
    ncpus: int
        the number of cpus that pool has access to
    '''
    full_df = full_df.set_index('sequence_id')
    
    #find possible dist param sets
    dist_param_list = list(set((tuple(params) for params in dist_param_map.values())))
    
    score_df = [] #to store scores
    
    #calculate silhouette score for each clustering
    for id_col in tqdm(id_cols):
        
        N_full = len(full_df)
        N_fam = len(full_df[id_col].unique())
        
        if N_fam == 1: #edge case for one giant family
            internal = np.random.randint(N_full,size=[sample_N_pairs,2])
            internal_dists = pool.starmap(partial(weighted_anarci_dist,
                                                  weight_scheme=weight_scheme,
                                                  max_len_diffs=max_len_diff),
                                          ([full_df.iloc[internal[i,0],:].loc[['CDR1','CDR2','CDR3']].tolist(),full_df.iloc[internal[i,1],:].loc[['CDR1','CDR2','CDR3']].tolist()] for i in range(sample_N_pairs)))
            external_dists = 0
            
        elif N_fam == N_full: #edge case for all separate families
            external = np.random.randint(N_full,size=[sample_N_pairs,2])
            external_dists = pool.starmap(partial(weighted_anarci_dist,
                                                  weight_scheme=weight_scheme,
                                                  max_len_diffs=max_len_diff),
                                          ([full_df.iloc[external[i,0],:].loc[['CDR1','CDR2','CDR3']].tolist(),full_df.iloc[external[i,1],:].loc[['CDR1','CDR2','CDR3']].tolist()] for i in range(sample_N_pairs)))
            internal_dists = 1
            
        else: #main case - sample randim pairs untul there are enough of each type
            N_ext = 0
            N_int = 0
            internal = np.zeros([sample_N_pairs,2])
            external = np.zeros([sample_N_pairs,2])
            while N_ext < sample_N_pairs:
                sample = np.random.randint(N_full,size=[sample_N_pairs,2])
                match = full_df[id_col].iloc[sample[:,0]].values == full_df[id_col].iloc[sample[:,1]].values
                
                N_match = match.sum()
                N_nonmatch = sample_N_pairs - N_match
                
                internal[min(N_int,sample_N_pairs):min(N_int+N_match,sample_N_pairs),:] = sample[match,:][:max(0,min(N_match,sample_N_pairs-N_int)),:]
                external[min(N_ext,sample_N_pairs):min(N_ext+N_nonmatch,sample_N_pairs),:] = sample[~match,:][:max(0,min(N_nonmatch,sample_N_pairs-N_ext)),:]
                
                N_int += N_match
                N_ext += N_nonmatch
                
            while N_int < sample_N_pairs:
                sample = np.random.randint(N_full,size=[sample_N_pairs,2])
                match = full_df[id_col].iloc[sample[:,0]].values == full_df[id_col].iloc[sample[:,1]].values
                
                N_match = match.sum()
                internal[N_int:min(N_int+N_match,sample_N_pairs),:] = sample[match,:][:min(N_match,sample_N_pairs-N_int),:]
                N_int += N_match
            
            internal = internal.astype(int)
            external = external.astype(int)
                
            weight_scheme = [1,1,dist_param_map[id_col][0]]
            max_len_diff = [dist_param_map[id_col][1]]*3
            
            internal_dists = pool.starmap(partial(weighted_anarci_dist,
                                                    weight_scheme=weight_scheme,
                                                    max_len_diffs=max_len_diff),
                                            ([full_df.iloc[internal[i,0],:].loc[['CDR1','CDR2','CDR3']].tolist(),full_df.iloc[internal[i,1],:].loc[['CDR1','CDR2','CDR3']].tolist()] for i in range(sample_N_pairs)))
                                          
            external_dists = pool.starmap(partial(weighted_anarci_dist,
                                                    weight_scheme=weight_scheme,
                                                    max_len_diffs=max_len_diff),
                                            ([full_df.iloc[external[i,0],:].loc[['CDR1','CDR2','CDR3']].tolist(),full_df.iloc[external[i,1],:].loc[['CDR1','CDR2','CDR3']].tolist()] for i in range(sample_N_pairs)))
        
        score = np.mean(external_dists) - np.mean(internal_dists)

        score_df.append([id_col,score])
        print(f'Score: {id_col}, {score}',flush=True)
    
    return pd.DataFrame(data=score_df,columns=['name','score'])

if __name__ == '__main__':
    
    args = get_args()
    
    #set up for parallelization
    ncpus = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
    pool = Pool(processes=ncpus)
    print(f'Connected to pool with {ncpus} cpus.',flush=True)
    
    #read in file and get rid of any duplicate entries
    data = pd.read_csv(args.in_file)
    if args.drop_duplicates != 'none':
        data = data.drop_duplicates(args.drop_duplicates)
        
    #get ANARCI parses if they are not present
    if 'CDR3' not in data.columns:
        anarci_info = pd.DataFrame(pool.map(get_anarci_alignment, data['VHH'], chunksize=1000),
                                   columns=['VHH', 'anarci_parse', 'CDR1', 'CDR2', 'CDR3', 'ANARCI_success'])
        data = data.merge(anarci_info, on='VHH', how='left')

    #filter out any sequences with missing CDR info
    data = data[data[['CDR1','CDR2','CDR3']].notna().all(axis=1)]
    
    tmp_idx = np.random.randint(1000000)
    if '/' not in args.out_file:
        tmp_dir = f'./tmp_{tmp_idx}'
    else:
        tmp_dir = '/'.join(args.out_file.split('/')[:-1])+f'/tmp_{tmp_idx}'
    
    dist_param_map = {col:(args.CDR3_weight,args.max_len_diff) for col in args.label_columns}
    
    score_df = weighted_ANARCI_silhouette_fast(data, args.label_columns, dist_param_map, args.sample_N_pairs, pool, ncpus)
    score_df.to_csv(args.out_file)