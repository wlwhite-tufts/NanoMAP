import pandas as pd
import numpy as np
import sys
from tqdm import tqdm
import os
import subprocess
import argparse
from scipy.cluster.hierarchy import linkage, fcluster, leaves_list
from multiprocessing import Pool

#get main dir for repo
repo_path = os.path.dirname(os.path.abspath(__file__))
repo_path = '/'.join(repo_path.split('/')[:-1])
sys.path.append(repo_path)
from utils.utils import *

def get_args():
    #get user inputs
    parser = argparse.ArgumentParser(prog='Final Tabulation',
                                     description='Collect enrichment and other info for clustered sequences.')
    #main args
    parser.add_argument('--in_file',type=str,help='The file that has the family and annotation information for all sequences in the experiment.')
    parser.add_argument('--sample_file',type=str,help='The csv file containig the names and file paths for all samples to be analyzed.')
    parser.add_argument('--pair_file',type=str,help='The csv file denoting which gorups of samples should be compared to each other for enrichment analysis.')
    parser.add_argument('--suffix',type=str,default='',help='Additional suffix for output csv files.')
    parser.add_argument('--seq_col',type=str,default='VHH',help='The column name to use for identifying unique sequences.')
    parser.add_argument('--fam_col',type=str,default='meta_clone_id_single_0.25',help='The column name to use for identifying families of sequences.')
    parser.add_argument('--gfold_quant',type=float,default=[0.01],nargs='*',help='The folder to store the preseq analysis results in if requested.')
    parser.add_argument('--header',type=str,default=[],nargs='*',help='The column names from sample_file to add as header descriptions to each of the columns.')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    
    GS_file = os.path.realpath(args.in_file) #input global set file
    samples = args.sample_file #file describing which samples to load and what names to give them
    pairs = args.pair_file #file describing which samples to use for enrichment analysis
    
    ####################
    # Collect data from all experiments
    ####################
    
    #read in sample names and enrichment pairs for later
    sample_df = pd.read_csv(samples)
    pair_df = pd.read_csv(pairs)
    
    #get list of indivisual experiment files to merge on
    print('Samples to load:')
    print(sample_df)
        
    #get global set with cluster info
    global_set = pd.read_csv(GS_file,index_col=0,dtype={args.fam_col:str})
    
    #make sure clone_id is a string not a numeric
    #otherwise, for some reason sometimes the same id can be read in as a string or an int
    global_set[args.fam_col] = global_set[args.fam_col].astype(str)
    
    #merge on individual experiment counts
    for i,row in tqdm(sample_df.iterrows()):
        # print(row['name'], row['file'], flush=True)
        tsv_data = pd.read_csv(row['file'],sep='\t')[['sequence','duplicate_count']]
        # print(tsv_data['duplicate_count'].sum(), flush=True)
        tsv_data = tsv_data.rename(columns={'duplicate_count':row['name']})
        # print(tsv_data[row['name']].sum(), flush=True)
    
        global_set = global_set.merge(tsv_data,on='sequence',how='left')
        # print(global_set[row['name']].sum(), flush=True)
        global_set.loc[np.isnan(global_set[row['name']]),row['name']] = 0 #set NaNs to 0
        # print(global_set[row['name']].sum(), flush=True)
        # print('-'*50)
        
    #sort by VHH_duplicate_count for later
    global_set = global_set.sort_values(by=['VHH_duplicate_count','duplicate_count'],ignore_index=True,ascending=False) #highest first
    
    #save global set with count info
    global_set.to_csv(GS_file.replace('.csv','_tabulated.csv'),index=False)
            
    ####################
    # calculate fold changes based on VHH-aggregated or clone_id-aggregated data
    ####################
    
    #connect to parallel pool
    ncpus = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
    pool = Pool(processes=ncpus)
    print(f'Connected to pool with {ncpus} cpus.',flush=True)
    
    #define count columns and aggregation method
    count_names = set(list(sample_df['name'].unique()))
    
    agg_funcs = {}
    for c in global_set.columns:
        #take the first of each VHH group for descriptor columns
        if c not in count_names:
            agg_funcs[c] = lambda x: x.iloc[0]
        #sum the duplicate count
        else:
            agg_funcs[c] = np.sum
    
    vhh = aggregate_and_calculate_fc(global_set,agg_funcs,args.seq_col,pair_df,args.gfold_quant,pool)
    fam = aggregate_and_calculate_fc(global_set,agg_funcs,args.fam_col,pair_df,args.gfold_quant,pool)
    
    ####################
    # add header if requested
    ####################
    
    if len(args.header):
        
        vhh = add_header(vhh, args.header, sample_df)
        fam = add_header(fam, args.header, sample_df)
    
    base_dir = '/'.join(GS_file.split('/')[:-1])
    base_name = GS_file.split('/')[-1].replace('_final_clusters','').replace('_clust','').replace('_metaclustering','').replace('.csv','')
    vhh.to_csv(f'{base_dir}/{base_name}{args.suffix}_gfold_vhh.csv',index=len(args.header)>0)
    fam.to_csv(f'{base_dir}/{base_name}{args.suffix}_gfold_fam.csv',index=len(args.header)>0)