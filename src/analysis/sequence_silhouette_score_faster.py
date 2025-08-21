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
import dask.dataframe as dd

sys.path.append('/cluster/tufts/cowenlab/wwhite06/packages/VHH-clustering/')
from src.utils.python_utils import weighted_anarci_dist, get_anarci_alignment

#get user inputs
parser = argparse.ArgumentParser(prog='Sequence-based evaluation of clutering quality',
                                 description='Calculate the silhouette score based on CDR distance for a given (set of) clustering(s)..')
#I/O args
parser.add_argument('--in_file',type=str,help='csv file containing sequence and cluster label info')
parser.add_argument('--out_file',type=str,help='Name for the output file (can include file path).')
parser.add_argument('--label_columns',type=str,nargs='*',help='Which column(s) contain the cluster labels to be evaluated')
parser.add_argument('--drop_duplicates',type=str,default='VHH',help='Which column to use to drop duplicate entries before scoring. Set to "none" to skip this step.')


#distance function args
parser.add_argument('--CDR3_weight',type=float,default=3.3, help='Weight for CDR3 in weighted distance calculations. Uses a weight of 1 for CDR1 and CDR2.')
parser.add_argument('--max_len_diff',type=int,default=3, help='Maximum allowed difference in CDR length. Distance is set to 1 if this is exceeded.')

#scoring args
parser.add_argument('--sample_N_pairs',type=int,default=10000,help='When calculating intra-family and inter-family distances, calcualte weighted distances for a random sampling of sample_N_pairs pairs of sequences.')

def weighted_ANARCI_silhouette(full_df, id_cols, dist_param_map, sample_N_pairs, pool, ncpus):
    
    full_df = full_df.copy().sort_values(by='VHH_duplicate_count',ascending=False)
    fill_df = full_df.set_index('sequence_id')
    full_df = dd.from_pandas(full_df,sort=False)
    
    #find possible dist param sets
    dist_param_list = list(set((tuple(params) for params in dist_param_map.values())))
    
    score_df = [] #to store scores
    
    #calculate silhouette score for each clustering
    for id_col in tqdm(id_cols):
        
        #get first sample in each pair to calculate dists
        internal = full_df.sample(sample_N_pairs,replace=True)
        internal['partner'] = ''
        external = full_df.sample(sample_N_pairs,replace=True)
        external['partner'] = ''
        
        #get second sample in each pair
        groups = []
        for g,group in internal.groupby(id_col):
            group['partner'] = full_df[full_df[id_col]==g].sample(len(group),replace=True).index
            groups.append(group)
        internal = pd.concat(groups,axis=0)
        
        groups = []
        for g,group in external.groupby(id_col):
            group['partner'] = full_df[full_df[id_col]!=g].sample(len(group),replace=True).index
            groups.append(group)
        external = pd.concat(groups,axis=0)
            
        weight_scheme = [1,1,dist_param_map[id_col][0]]
        max_len_diff = [dist_param_map[id_col][1]]*3
        
        internal['dist'] = pool.starmap(partial(weighted_anarci_dist,
                                                weight_scheme=weight_scheme,
                                                max_len_diffs=max_len_diff),
                                        ([row[['CDR1','CDR2','CDR3']].tolist(),full_df.loc[row['partner'],['CDR1','CDR2','CDR3']].tolist()] for _,row in internal.iterrows()))
                                      
        external['dist'] = pool.starmap(partial(weighted_anarci_dist,
                                                weight_scheme=weight_scheme,
                                                max_len_diffs=max_len_diff),
                                        ([row[['CDR1','CDR2','CDR3']].tolist(),full_df.loc[row['partner'],['CDR1','CDR2','CDR3']].tolist()] for _,row in external.iterrows()))
        
        score = external['dist'].mean() - internal['dist'].mean()

        score_df.append([id_col,score])
        print(f'Score: {id_col}, {score}',flush=True)
    
    return pd.DataFrame(data=score_df,columns=['name','score'])

if __name__ == '__main__':
    
    args = parser.parse_args()
    
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
    
    score_df = weighted_ANARCI_silhouette(data, args.label_columns, dist_param_map, args.sample_N_pairs, pool, ncpus)
    score_df.to_csv(args.out_file)