import numpy as np
import pandas as pd
import sys
import os
import subprocess
from sklearn.cluster import AgglomerativeClustering
from multiprocessing import Pool
from glob import glob
from tqdm import tqdm
import argparse
import itertools
from collections import defaultdict
from scipy.cluster.hierarchy import linkage, fcluster
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.append('/cluster/tufts/cowenlab/wwhite06/packages/VHH-clustering/src/')
from utils.python_utils import *
from clustering.anarci.meta_ANARCI_clustering_alt import *

#get user inputs
parser = argparse.ArgumentParser(prog='Sequential clustering',
                                 description='Sequential clustering of time points based on ANARCI clustering and closest matching.')
#I/O args
parser.add_argument('--in_file_list',type=str,help='The file that contains the list of groups of files to cluster. One group per line, separated by commas. Clustering will proceed in the order that groups appear in the file.')
parser.add_argument('--out_file',type=str,help='Name for the output file (can include file path).')

#translation args
parser.add_argument('--start_seq',type=str,default='LVQPGGSLRLSC',help='Amino acid sequence to scan for when finding starting point of translation.')
parser.add_argument('--end_seq',type=str,default='WGQGTQVTVSS',help='Amino acid sequence to scan for when finding ending point of translation.')
parser.add_argument('--min_start_match',type=int,default=5,help='Minimum number of amino acids matching start_seq to be considered a real sequence.')
parser.add_argument('--min_end_match',type=int,default=5,help='Minimum number of amino acids matching end_seq to be considered a real sequence.')
parser.add_argument('--max_start_scan',type=int,default=40,help='Number of N-term amino acid positions to scan when looking for start_seq.')
parser.add_argument('--max_end_scan',type=int,default=40,help='Number of C-term amino acid positions to scan when looking for end_seq.')

#clustering args
parser.add_argument('--max_subgroup_size',type=int,default=25000,help='Maximum number of VHHs in a SCOPer (sub)group before triggering a recursive loose mmseqs sub-clustering.')
parser.add_argument('--recursive_min_id_start',type=float,nargs=2,default=[0.5,0.8],help='Inclusive bounds for range of starting values for minimum sequence identity allowed for recursive loose mmseqs clustering.')
parser.add_argument('--recursive_min_id_start_N',type=int,default=5,help='Number of values to test between the recursive_min_id_start bounds.')
parser.add_argument('--recursive_min_id_step',type=float,default=0.025,help='Step size value for minimum sequence identity incerment in each recursive layer.')

parser.add_argument('--recursive_method',type=str,nargs='*',default=['setcover'],help='Method to use for clustering in the recursive mmseqs clustering (setcover or conncomp).')

parser.add_argument('--CDR3_weight',type=float,nargs=2,default=[3,3], help='Inclusive bounds for the weight for CDR3 in weighted distance calculations. Uses a weight of 1 for CDR1 and CDR2.')
parser.add_argument('--CDR3_weight_N',type=int,default=1,help='Number of values to test between the CDR3_weight bounds.')

parser.add_argument('--max_len_diff',type=int,nargs=2,default=[3,3], help='Inclusive bounds for the range of values of maximum allowed difference in CDR length.')
parser.add_argument('--max_len_diff_N',type=int,default=1,help='Number of values to test between the max_len_diff bounds.')

parser.add_argument('--len_diff_penalty',type=float,nargs=2,default=[0.01,0.05], help='Inclusive bounds for the range of values of the distance penalty for CDR length differences.')
parser.add_argument('--len_diff_penalty_N',type=int,default=3,help='Number of values to test between the len_diff_penalty bounds.')

parser.add_argument('--linkage',type=str,default=['single','average','complete'],nargs='+',help='The linkage methods to try in weighted hierarchical clustering step (can be single, complete, or average).')

parser.add_argument('--weighted_min_id',type=float,nargs=2,default=[0.4,0.8],help='Inclusinve bounds for the range of distance cutoff values to use in the weighted hierarchical clustering step.')
parser.add_argument('--weighted_min_id_N',type=int,default=5,help='Number of values to test between the weighted_min_id bounds.')

parser.add_argument('--score_fraction',type=float,default=0.5,help='Keep the top score_fraction of individual clusterings. Only these clusterings are used in metaclustering.')
parser.add_argument('--dist_cutoff',type=float,default=0.4,help='CDR3 distance cutoff to use for determining if a pair of sequences within a cluster matches closely enough when scoring.')

parser.add_argument('--meta_min_id',type=float,default=[0.45],nargs='*',help='Minimum fraction of agreeing clusterings to group two VHHs. If given multiple values will try all. Clustering distance threshhold = 1 - meta_min_id')
parser.add_argument('--meta_method',type=str,default=['single'],nargs='*',help='Linkage method to use for meta clustering (can be single, average, or complete). If given multiple values will try all.')

#linking args
parser.add_argument('--min_link_id',type=float,default=0.9,help='Minimum fractional identity required to link a sequence from one group to the previously clustered sequences.')
parser.add_argument('--figure_file',type=str,default='',help='File name prefix for saving nearest neighbor distance distributions for each linking step. Files are not saved if figure_file is blank.')

def find_closest_match(query_df,target_df,tmp_dir,figure_fname):
    os.mkdir(tmp_dir)
    
    #write temp fasta files
    with open(f'{tmp_dir}tmp_mmseqs_query.fasta','w') as f:
        for i,row in query_df.iterrows():
            f.write(f'>{row.name}\n{row["VHH"]}\n')
    
    with open(f'{tmp_dir}tmp_mmseqs_target.fasta','w') as f:
        for i,row in target_df.iterrows():
            f.write(f'>{row.name}\n{row["VHH"]}\n')
            
    #run mmseqs to find closest family
    result = subprocess.run(['singularity', 'exec', '/cluster/tufts/biocontainers/images/quay.io_biocontainers_mmseqs2:17.b804f--hd6d6fdc_1.sif', 'mmseqs',
                             'easy-search', f'{tmp_dir}tmp_mmseqs_query.fasta', f'{tmp_dir}tmp_mmseqs_target.fasta', f'{tmp_dir}/mmseqs_results', tmp_dir,
                             '-s', '7.5', #sensitivity
                             '-c', '0', #don't filter on coverage
                             '--min-seq-id', '0', #don't filter here because we want to have full distribution info
                             '--local-tmp', tmp_dir,
                             '--format-output', 'query,target,pident',
                             '--max-seq-id','1.0', #do not do any redundancy filtering
                             '--max-seqs', '1', #look at best seq for each query
                             '-v', '0'])
    assert result.returncode == 0
                    
    mmseqs_df = pd.read_csv(f'{tmp_dir}/mmseqs_results',names=['query','target','pident'],sep='\t')
    
    if len(figure_fname):
        plt.figure(figsize=[8,8])
        sns.histplot(mmseqs_df['pident'],bins=np.linspace(0,100,100))
        plt.xlabel('Identity to Closest Match',fontsize=24)
        plt.ylabel('Number of Sequences',fontsize=24)
        plt.savefig(figure_fname,dpi=300,bbox_inches='tight')
        
    subprocess.run(['rm', '-r', tmp_dir])
        
    return mmseqs_df

if __name__ == '__main__':
    args = parser.parse_args()
    
    #set up for parallelization
    ncpus = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
    pool = Pool(processes=ncpus)
    print(f'Connected to pool with {ncpus} cpus.',flush=True)
    
    #make tmp_dir
    out_dir = '/'.join(args.out_file.split('/')[:-1])+'/'
    out_fname = args.out_file.split('/')[-1]
    tmp_dir = f'{out_dir}tmp/'

    file_groups = [line.strip().split(',') for line in open(args.in_file_list,'r').readlines()]
    
    #get basic info for all seqs
    data = []
    for i,group in enumerate(file_groups):
        group_df = collect_seqs(group)
        group_df[f'group{i}'] = True
        data.append(group_df)
    data = pd.concat(data,axis=0,ignore_index=True)

    #define functions for aggregation
    agg_funcs = {}
    for c in data.columns:
        if c == 'duplicate_count':
            agg_funcs[c] = np.sum #add up duplicate counts
        elif 'group' in c: #take logical "any" for group labels
            agg_funcs[c] = np.any
        else: #take the first of each VHH group for descriptor columns
            agg_funcs[c] = lambda x: x.iloc[0]

    #get translations, CDR anotations, and filter out bad/singleton sequences across whole set
    data,vhh = annotate_and_filter_seqs(args,data,pool,agg_funcs)
    
    #get clustering for first group
    vhh_new,_,meta_id_cols = meta_ANARCI_clustering_alt(args,vhh[vhh['group0']].copy(),out_dir,out_fname,pool,ncpus)
    keep_cols = [col for col in vhh_new if ('clone_id' not in col) and (col != 'scoper_group')]
    vhh_new = vhh_new[keep_cols+meta_id_cols] #remove unnecessary columns
    vhh_new = vhh_new.rename(columns={f'meta_clone_id_{args.meta_method[0]}_{args.meta_min_id[0]}':'sequential_clone_id'}) #rename meta id col
    vhh = pd.concat([vhh.loc[~vhh['group0'],:].copy(),vhh_new.copy()],axis=0,join='outer') #save cluster labels in main vhh df
    
    #set index to sequence_id so that it's easy to assign new labels
    vhh = vhh.set_index('sequence_id')
    
    #cluster new members added in each group, one group at a time
    for i in range(1,len(file_groups)):
        
        #find idxs that have already been taken care of
        old_idx = vhh[[f'group{ii}' for ii in range(i)]].any(axis=1)
        
        missing = (old_idx&(vhh['sequential_clone_id'].isna())).sum()
        print(f'Pre i = {i}: {missing} seqeunces are missing labels out of {old_idx.sum()} that should be.',flush=True)
        
        #find idxs in this group but not any previous groups
        new_idx = vhh[f'group{i}'] & ~old_idx
        print(f'{new_idx.sum()} new sequences added at step {i}',flush=True)
        
        # create figure file name if requested
        if len(args.figure_file):
            fig_file = f'{args.figure_file}_{i+1}.png'
        else:
            fig_file = ''
        
        #find matches for new sequences
        matches = find_closest_match(vhh[new_idx].copy(),vhh[old_idx].copy(),tmp_dir,f'{args.figure_file}_{i}.png')
        
        #split into good and bad matches, apply matched cluster labels
        good_matches = matches.loc[matches['pident']>=args.min_link_id*100,:]
        
        vhh.loc[good_matches['query'],'sequential_clone_id'] = vhh.loc[good_matches['target'],'sequential_clone_id'].copy().tolist()
        print(f'{len(good_matches)} of the new sequences added at step {i} have good matches (>= {args.min_link_id*100}%ID)',flush=True)
        
        #cluster any unmatched new sequences (mmseqs doesn't always give a result for everything, so we have to go back to the original list of new seqs)
        unmatched = vhh[new_idx].drop(good_matches['query'],axis=0)
        print(f'{len(unmatched)} sequences need to be clustered at step {i}.',flush=True)
        unmatched,_,meta_id_cols = meta_ANARCI_clustering_alt(args,unmatched,out_dir,out_fname,pool,ncpus)
        unmatched = unmatched[keep_cols+meta_id_cols] #remove unnecessary columns
        unmatched = unmatched.rename(columns={f'meta_clone_id_{args.meta_method[0]}_{args.meta_min_id[0]}':'sequential_clone_id'}) #rename meta id col
        unmatched = unmatched.set_index('sequence_id') #set index back to sequence_id to match vhh dataframe

        #apply cluster labels to newly clustered sequences
        min_clust = vhh['sequential_clone_id'].max() + 1
        unmatched['sequential_clone_id'] += min_clust
        not_unmatched = vhh.drop(unmatched.index,axis=0)
        vhh = pd.concat([unmatched,not_unmatched],axis=0)
    
    print(f'Post: {(vhh["sequential_clone_id"].isna()).sum()} seqeunces are missing labels out of {len(vhh)} that should be.',flush=True)
    print(f'{len(vhh["sequential_clone_id"].unique())} families found after linking.',flush=True)
    
    #merge results back onto data
    print(f'{len(data)} rows before merge.',flush=True)
    data = data.merge(vhh[['VHH','CDR1','CDR2','CDR3','sequential_clone_id']],on='VHH',how='outer')
    print(f'{len(data)} rows after merge.',flush=True)
        
    #save results
    data.to_csv(args.out_file)
        
