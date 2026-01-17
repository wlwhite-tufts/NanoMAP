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

#get main dir for repo
repo_path = os.path.dirname(os.path.abspath(__file__))
repo_path = '/'.join(repo_path.split('/')[:-1])
sys.path.append(repo_path)
from utils import weighted_anarci_dist, ANARCI_dist_row, get_fam_reps, fill_in_weighted_anarci_dists, approx_avg_internal_anarci_dist, \
                  get_pairwise_cluster_rep_distances, agg_clust_to_max_extra_reps, get_anarci_alignment
from data.user_data import mmseqs_sif

def get_args():
    #get user inputs
    parser = argparse.ArgumentParser(prog='Sequence-based evaluation of clutering quality',
                                     description='Calculate the silhouette score based on CDR distance for a given (set of) clustering(s)..')
    #I/O args
    parser.add_argument('--in_file',type=str,help='csv file containing sequence and cluster label info')
    parser.add_argument('--out_file',type=str,help='Name for the output file (can include file path).')
    parser.add_argument('--label_columns',type=str,nargs='*',help='Which column(s) contain the cluster labels to be evaluated')
    parser.add_argument('--drop_duplicates',type=str,default='VHH',help='Which column to use to drop duplicate entries before scoring. Set to "none" to skip this step.')
    
    
    #distance function args
    parser.add_argument('--CDR3_weight',type=float,default=3, help='Weight for CDR3 in weighted distance calculations. Uses a weight of 1 for CDR1 and CDR2.')
    parser.add_argument('--max_len_diff',type=int,default=3, help='Maximum allowed difference in CDR length. Distance is set to 1 if this is exceeded.')
    
    #scoring args
    parser.add_argument('--score_top_N',type=int,default=10,help='When calculating inter-family distances, calcualte weighted distances to the top score_top_N MMseqs hits for each family representative.')
    parser.add_argument('--min_clust_reps',type=int,default=10,help='When calculating inter-family distances, select at least the top min_clust_reps most abundant members to represent each family.')
    parser.add_argument('--max_extra_reps',type=int,default=150,help='When calculating inter-family distances, split clusterings into groups of no more than max_group_size members based on the overlap of their representatives.')
    parser.add_argument('--sample_N_pairs',type=int,default=1000,help='When calculating intra-family distances, calcualte weighted distances for a random sampling of sample_N_pairs pairs of sequences within the cluster.')
    
    return parser.parse_args()
    
def weighted_ANARCI_silhouette(full_df, id_cols, mmseqs_n, min_n_reps, max_extra_reps, dist_param_map, sample_N_pairs, pool, ncpus, tmp_dir):
    
    full_df = full_df.copy()
    
    #get groupings that are relatively similar to each other
    clust_dists = get_pairwise_cluster_rep_distances(full_df,id_cols,min_n_reps)
    labels = agg_clust_to_max_extra_reps(full_df,id_cols,clust_dists,'complete',min_n_reps,max_extra_reps)
    
    #find possible dist param sets
    dist_param_list = list(set((tuple(params) for params in dist_param_map.values())))
    
    #pool mmseqs calculations for groups of similar clustering
    score_df = []
    for c in np.unique(labels):
        label_match = labels==c
        id_col_subset = [col for i,col in enumerate(id_cols) if label_match[i]]
        
        all_reps, max_reps = get_fam_reps(id_col_subset,full_df,min_n_reps)
        print(f'Collected representatives from all clusterings in group {c}. Total rows: {len(all_reps)}',flush=True)
        print(f'Across all {len(id_col_subset)} clusterigns in group {c}, the most represetned cluster has {max_reps} representatives (min value was set to {min_n_reps}).',flush=True)
        
        #run MMseqs to find close matches across family representaives
        os.mkdir(tmp_dir)
        with open(f'{tmp_dir}/tmp_silhouette_mmseqs.fasta','w') as f:
            for i,row in all_reps.iterrows():
                f.write(f'>{row.name}\n{row["VHH"]}\n')
                
        result = subprocess.run(['singularity', 'exec', mmseqs_sif, 'mmseqs', 'easy-search',
                                f'{tmp_dir}/tmp_silhouette_mmseqs.fasta', f'{tmp_dir}/tmp_silhouette_mmseqs.fasta', f'{tmp_dir}/mmseqs_results', tmp_dir,
                                 '-s', '7.5', #sensitivity
                                 '-c', '0', #don't filter on coverage
                                 '--min-seq-id', '0', #don't filter here because we want to have full distribution info
                                 '--local-tmp', tmp_dir,
                                 '--format-output', 'query,target,pident',
                                 '--max-seq-id','1.0', #do not do any redundancy filtering
                                 '--max-seqs', str(mmseqs_n+max_reps), #look at best seq for each query (add min_n_reps to account for finding other sequences representing the same cluster)
                                 '-v', '0',
                                 '--threads', str(ncpus)])
        assert result.returncode == 0
                        
        mmseqs_df = pd.read_csv(f'{tmp_dir}/mmseqs_results',names=['query','target','pident'],sep='\t')
        
        #remove tmp files
        subprocess.run(['rm', '-r', tmp_dir])
        
        #remove matches to self
        mmseqs_df = mmseqs_df[mmseqs_df['query']!=mmseqs_df['target']]
        
        #merge on clustering info about each query and target
        mmseqs_df = mmseqs_df.merge(full_df[['sequence_id']+id_col_subset],left_on='query',right_on='sequence_id',how='left')
        mmseqs_df = mmseqs_df.merge(full_df[['sequence_id']+id_col_subset],left_on='target',right_on='sequence_id',how='left',suffixes=('_query','_target'))
        
        #add columns for CDR distances
        dist_idx_map = {}
        for i,params in enumerate(dist_param_list):
            mmseqs_df[f'dist_{i}'] = np.nan
            dist_idx_map[params] = i
                                                                
        #calculate silhouette score for each clustering
        for id_col in tqdm(id_col_subset):
            
            #get rid of pairs within the same cluster
            mmseqs_subset = mmseqs_df[mmseqs_df[id_col+'_query']!=mmseqs_df[id_col+'_target']]
            
            #take top mmseqs_n MMseqs scores for each query
            mmseqs_subset = mmseqs_subset.sort_values(by='pident',ascending=False).groupby('query').head(mmseqs_n)
            
            #calculate dists as needed
            dist_idx = dist_idx_map[dist_param_map[id_col]]
            weight_scheme = [1,1,dist_param_map[id_col][0]]
            max_len_diffs = [dist_param_map[id_col][1]]*3
            mmseqs_subset,_ = fill_in_weighted_anarci_dists(mmseqs_subset, all_reps, f'dist_{dist_idx}', weight_scheme, max_len_diffs, pool, ncpus)
            
            #add subset dists to mmseqs_df
            mmseqs_df.loc[mmseqs_subset.index,f'dist_{dist_idx}'] = mmseqs_subset[f'dist_{dist_idx}']
            
            #calculate mean closest neighbor across all family representatives for each family
            external_dists = mmseqs_subset.loc[mmseqs_subset.groupby('query')[f'dist_{dist_idx}'].idxmin().tolist(),:].groupby(id_col+'_query')[f'dist_{dist_idx}'].mean()
            external_dists = external_dists.reset_index().rename(columns={f'dist_{dist_idx}':'external_dist',id_col+'_query':id_col})
    
            #calculate approximate average internal distances
            internal_dists = full_df.groupby(id_col).apply(lambda group: approx_avg_internal_anarci_dist(group,sample_N_pairs,weight_scheme,max_len_diffs,pool))
            internal_dists = internal_dists.reset_index().rename(columns={0:'internal_dist'})
            
            #get cluster sizes
            sizes = full_df.groupby(id_col).size().reset_index().rename(columns={0:'size'})
            
            #merge cluster info together
            info = sizes.merge(internal_dists,on=id_col,how='left').merge(external_dists,on=id_col,how='left')
            
            #fill in any missing values        
            missing_idx = info['external_dist'].isna()
            if missing_idx.sum():
                print(f'WARNING: {missing_idx.sum()} sequences in {id_col} are missing nearest external dist info - filling in with dist=1.', flush=True)
                info.loc[missing_idx,'external_dist'] = 1
            
            #score = nearest_external_dist - farthest_internal_dist
            score = ((info['external_dist'] - info['internal_dist'])*info['size']).sum()/info['size'].sum()

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
    
    score_df = weighted_ANARCI_silhouette(data, args.label_columns, args.score_top_N, args.min_clust_reps, args.max_extra_reps, dist_param_map, args.sample_N_pairs, pool, ncpus, tmp_dir)
    score_df.to_csv(args.out_file)