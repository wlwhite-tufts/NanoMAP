# current version of the script for calculating enrichment statistics
# requres that data are already clustered
# will need to be updated for compatibility with new clustering methods

import pandas as pd
import numpy as np
import sys
from tqdm import tqdm
import os
import subprocess
import argparse
from scipy.cluster.hierarchy import linkage, fcluster, leaves_list
from multiprocessing import Pool

sys.path.append('/cluster/tufts/cowenlab/wwhite06/packages/VHH-clustering/src/utils/')
from python_utils import *

#get user inputs
parser = argparse.ArgumentParser(prog='Final Tabulation',
                                 description='Collect enrichment and other info for clustered sequences.')
#main args
parser.add_argument('--GS_file',type=str,help='The file that has the family and annotation information for all sequences in the experiment.')
parser.add_argument('--suffix',type=str,default='',help='Additional suffix for output csv files.')
parser.add_argument('--sample_file',type=str,help='The csv file containig the names and file paths for all samples to be analyzed.')
parser.add_argument('--pair_file',type=str,help='The csv file denoting which gorups of samples should be compared to each other for enrichment analysis.')
parser.add_argument('--seq_col',type=str,default='VHH',help='The column name to use for identifying unique sequences.')
parser.add_argument('--fam_col',type=str,default='clone_id',help='The column name to use for identifying families of sequences.')
parser.add_argument('--preseq_dir',type=str,default='',help='The folder to store the preseq analysis results in if requested.')
parser.add_argument('--dendr_min_count',type=int,default= -1,help='The minimum number of reads in each experiment needed to be included in dendrogramming analysis.')
parser.add_argument('--dendr_linkage',type=str,default='average',help='The linkage type to use when creating the dendrogram order (complete, single, or average).')
parser.add_argument('--gfold_quant',type=float,default=[0.01],nargs='*',help='The folder to store the preseq analysis results in if requested.')
parser.add_argument('--header',type=str,default=[],nargs='*',help='The column names from sample_file to add as header descriptions to each of the columns.')

#args for preseq pre-processing
parser.add_argument('--start_seq',type=str,default='LVQPGGSLRLSC',help='Amino acid sequence to scan for when finding starting point of translation.')
parser.add_argument('--end_seq',type=str,default='WGQGTQVTVSS',help='Amino acid sequence to scan for when finding ending point of translation.')
parser.add_argument('--min_start_match',type=int,default=5,help='Minimum number of amino acids matching start_seq to be considered a real sequence.')
parser.add_argument('--min_end_match',type=int,default=5,help='Minimum number of amino acids matching end_seq to be considered a real sequence.')
parser.add_argument('--max_start_scan',type=int,default=40,help='Number of N-term amino acid positions to scan when looking for start_seq.')
parser.add_argument('--max_end_scan',type=int,default=40,help='Number of C-term amino acid positions to scan when looking for end_seq.')
parser.add_argument('--min_seq_id',type=int,default=90,help='Minimum percent identity to allow unclustered sequence to be assigned to a family.')

args = parser.parse_args()

GS_file = args.GS_file #input global set file
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
global_set.to_csv(GS_file.replace('_final_clusters',f'{args.suffix}_tabulated_seq').replace('_clust.csv',f'{args.suffix}_tabulated_seq.csv'),index=False)
        
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
# run dendrogramming if requested
####################
if args.dendr_min_count >= 0:
    
    os.mkdir('tmp')
    os.mkdir('tmp/fasta')
    os.mkdir('tmp/dist')
    
    #dendrogram for each sample
    for i,pair_row in pair_df.iterrows():
        #get subset that pass user-defined cutoff for this sample
        subset = fam.loc[fam[pair_row['final'].split('+')].sum(axis=1) >= args.dendr_min_count,:]
        
        print(f'Dendrogramming {len(subset)} families for {pair_row["name"]}.')
        
        #write fasta file
        fasta_name = f'tmp/fasta/{pair_row["name"]}.fasta'
        with open(fasta_name,'w') as f:
            for ii,sub_row in subset.iterrows():
                f.write(f">{sub_row[args.fam_col]}\n{sub_row[args.seq_col]}\n") #Index here is the intermediate_id
                
        #run mmseqs search/alignment
        dist_name = f'tmp/dist/{pair_row["name"]}_dists.tsv'
        result = subprocess.run(['/cluster/tufts/cowenlab/emosel01/condaenv/vhh/bin/mmseqs', 'easy-search', fasta_name, fasta_name, dist_name, 'tmp/tmp',
                                 '-s', '7.5', #sensitivity set to max
                                 '--min-seq-id', '0', #allow any amount of sequence matching
                                 '-c', '0', #allow any amount of sequence coverage
                                 '-e', 'inf', #max e-value to keep (set to keep all)
                                 '--min-ungapped-score', '0', #keep all
                                 '-v', '2', #verbosity
                                 '--seq-id-mode', '0', #use alignment length to normalize
                                 '--alignment-mode', '3', #get percent identity, rather than alignment score
                                 '--max-seqs', str(len(subset)+1), #get all matches to each query
                                 '--threads', '1']) #how many cpus to use
        assert result.returncode == 0
    
        dists = pd.read_csv(dist_name,
                            sep='\t',
                            names=['query','target','pident','alnlen','mismatch','gapopen','qstart','qend','tstart','tend','evalue','bits'],
                            dtype={'query':str,'target':str})
        #convert to wide-form
        dists = dists.pivot(index='query',columns='target',values='pident')
        print(f'{np.sum(np.isnan(dists.values))} missing distances of {len(subset)**2} total.',flush=True)
        #re-order to be consistent, convert to numpy array, convert from identity to dist
        dists = 1 - dists.loc[subset[args.fam_col],subset[args.fam_col]].values
        
        #get just upper triangular values for clustering
        dists = dists[np.triu_indices(dists.shape[0],k=1)]
        dists[np.isnan(dists)] = 1 #fill in missing values as 100% different from eachother
        
        print('Got dists, starting clustering.', flush=True)
        
        #get linkage and cluster
        links = linkage(dists, method=args.dendr_linkage, optimal_ordering=True)
        leaf_idx = leaves_list(links)

        #convert to df for merging
        labels = pd.DataFrame({args.fam_col:subset[args.fam_col], f'dendr_{pair_row["name"]}':leaf_idx})
        
        #merge labels onto main family df (will leave NaN values for families not passing cutoff)
        fam = fam.merge(labels,on=args.fam_col,how='left')
    
    subprocess.run(['rm', '-r', 'tmp'])

####################
# add header if requested
####################

if len(args.header):
    
    vhh = add_header(vhh, args.header, sample_df)
    fam = add_header(fam, args.header, sample_df)

vhh.to_csv(GS_file.replace('_final_clusters',f'{args.suffix}_gfold_vhh').replace('_clust.csv',f'{args.suffix}_gfold_vhh.csv'),index=len(args.header)>0)
fam.to_csv(GS_file.replace('_final_clusters',f'{args.suffix}_gfold_fam').replace('_clust.csv',f'{args.suffix}_gfold_fam.csv'),index=len(args.header)>0)

####################
# run preseq if requested
####################

if args.preseq_dir != '':
    
    os.mkdir(f'{args.preseq_dir}/tmp')
    with open(f'{args.preseq_dir}/tmp/mmseqs_db.fasta', 'w') as f:
        for i,row in vhh.iterrows():
            f.write(f'>{row["sequence_id"]}\n{row[args.seq_col]}\n')
    
    print('Running VHH-level preseq.',flush=True)
    run_preseq_assembly(data=global_set,
                        sample_df=sample_df,
                        pair_df=pair_df,
                        out_dir=args.preseq_dir,
                        kind='vhh',
                        pool=pool,
                        seq_col=args.seq_col,
                        fam_col=args.fam_col,
                        start_seq=args.start_seq,
                        end_seq=args.end_seq,
                        max_start_scan=args.max_start_scan,
                        max_end_scan=args.max_end_scan,
                        min_start_match=args.min_start_match,
                        min_end_match=args.min_end_match,
                        tmp_dir=f'{args.preseq_dir}/tmp',
                        mmseqs_db=f'{args.preseq_dir}/tmp/mmseqs_db.fasta',
                        min_seq_id=args.min_seq_id)
                        
    print('Running family-level preseq.',flush=True)
    run_preseq_assembly(data=global_set,
                        sample_df=sample_df,
                        pair_df=pair_df,
                        out_dir=args.preseq_dir,
                        kind='fam',
                        pool=pool,
                        seq_col=args.seq_col,
                        fam_col=args.fam_col,
                        start_seq=args.start_seq,
                        end_seq=args.end_seq,
                        max_start_scan=args.max_start_scan,
                        max_end_scan=args.max_end_scan,
                        min_start_match=args.min_start_match,
                        min_end_match=args.min_end_match,
                        tmp_dir=f'{args.preseq_dir}/tmp',
                        mmseqs_db=f'{args.preseq_dir}/tmp/mmseqs_db.fasta',
                        min_seq_id=args.min_seq_id)
                        
    #clean up
    result = subprocess.run(['rm', '-r', f'{args.preseq_dir}/tmp'])
    assert result.returncode == 0