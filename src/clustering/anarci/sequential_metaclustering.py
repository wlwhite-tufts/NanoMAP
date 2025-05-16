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
parser.add_argument('--figure_file',type='str',default='',help='File name prefix for saving nearest neighbor distance distributions for each linking step. Files are not saved if figure_file is blank.')

def find_closest_match(query_df,target_df,tmp_dir,args):
    #write temp fasta files
    with open(f'{tmp_dir}tmp_mmseqs_query.fasta','w') as f:
        for i,row in query_df.iterrows():
            f.write(f'>{row["sequence_id"]}\n{row["VHH"]}\n')
    
    with open(f'{tmp_dir}tmp_mmseqs_target.fasta','w') as f:
        for i,row in target_df.iterrows():
            f.write(f'>{row["sequence_id"]}\n{row["VHH"]}\n')
            
    #run mmseqs to find closest family
    result = subprocess.run(['/cluster/tufts/cowenlab/emosel01/condaenv/vhh/bin/mmseqs',
                             'easy-search', f'{tmp_dir}/tmp_mmseqs_query.fasta', mmseqs_db, f'{tmp_dir}/mmseqs_results', tmp_dir,
                             '-s', '7.5', #sensitivity
                             '-c', '0', #don't filter on coverage
                             '-min-seq-id', '0', #don't filtre here because we want to have full distribution info
                             '--local-tmp', tmp_dir,
                             '--format-output', 'query,target,pident',
                             '--max-seq-id','1.0', #do not do any redundancy filtering
                             '--max-seqs', '1', #look at best seq for each query
                             '-v', '0'])
    assert result.returncode == 0
                    
    mmseqs_df = pd.read_csv(f'{tmp_dir}/mmseqs_results',names=['query','target','pident'],sep='\t')
    
    if len(args.figure_file):
        plt.figure(figsize=[8,8])
        sns.histplot(mmseqs_df['pident'],bins=np.)

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

    file_groups = [line.split(',') for line in open(args.in_file_list,'r').readlines()]

    #cluster the first group from scratch
    data = collect_seqs(file_groups[0]) #get sequence info from user-specificed folder
    data,vhh = annotate_and_filter_seqs(args,data,pool) #get translations, CDR anotations, and filter out bad/singleton sequences
    vhh,_,_ = meta_ANARCI_clustering_alt(args,vhh,out_dir,out_fname,pool,ncpus) #get cluster labels

    


