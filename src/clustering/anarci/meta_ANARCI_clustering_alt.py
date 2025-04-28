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

sys.path.append('/cluster/tufts/cowenlab/wwhite06/packages/VHH-clustering/src/utils/')
from python_utils import *

#get user inputs
parser = argparse.ArgumentParser(prog='Cluster Based on ANARCI alignment',
                                 description='Collect sequences from input folder and cluster them using an agglomerative clustering method.')
#I/O args
parser.add_argument('--in_dir',type=str,help='The directory that contains the assembled sequences. Looks for *db-pass.tsv, or */*db-pass.tsv')
parser.add_argument('--out_file',type=str,help='Name for the output file (can include file path).')
parser.add_argument('--keep_intermediates',action='store_true',help='Set this flag to save the cluster IDs for each clustering leading to the metaclustering.')

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
parser.add_argument('--recursive_min_id_start_N',type=int,default=3,help='Number of values to test between the recursive_min_id_start bounds.')
parser.add_argument('--recursive_min_id_step',type=float,default=0.025,help='Step size value for minimum sequence identity incerment in each recursive layer.')

parser.add_argument('--CDR3_weight',type=float,nargs=2,default=[3.3,3.3], help='Inclusive bounds for the weight for CDR3 in weighted distance calculations. Uses a weight of 1 for CDR1 and CDR2.')
parser.add_argument('--CDR3_weight_N',type=int,default=1,help='Number of values to test between the CDR3_weight bounds.')

parser.add_argument('--max_len_diff',type=int,nargs=2,default=[3,3], help='Inclusive bounds for the range of values of maximum allowed difference in CDR length.')
parser.add_argument('--max_len_diff_N',type=int,default=1,help='Number of values to test between the max_len_diff bounds.')

parser.add_argument('--len_diff_penalty',type=float,nargs=2,default=[0.01,0.05], help='Inclusive bounds for the range of values of the distance penalty for CDR length differences.')
parser.add_argument('--len_diff_penalty_N',type=int,default=3,help='Number of values to test between the len_diff_penalty bounds.')

parser.add_argument('--linkage',type=str,default=['single','average','complete'],nargs='+',help='The linkage methods to try in weighted hierarchical clustering step (can be single, complete, or average).')

parser.add_argument('--weighted_min_id',type=float,nargs=2,default=[0.3,0.7],help='Inclusinve bounds for the range of distance cutoff values to use in the weighted hierarchical clustering step.')
parser.add_argument('--weighted_min_id_N',type=int,default=3,help='Number of values to test between the weighted_min_id bounds.')

parser.add_argument('--score_fraction',type=float,default=0.5,help='Keep the top score_fraction of individual clusterings. Only these clusterings are used in metaclustering.')
parser.add_argument('--dist_cutoff',type=float,default=0.4,help='CDR3 distance cutoff to use for determining if a pair of sequences within a cluster matches closely enough when scoring.')

parser.add_argument('--meta_min_id',type=float,default=[0.35],nargs='*',help='Minimum fraction of agreeing clusterings to group two VHHs. If given multiple values will try all. Clustering distance threshhold = 1 - meta_min_id')
parser.add_argument('--meta_method',type=str,default=['single'],nargs='*',help='Linkage method to use for meta clustering (can be single, average, or complete). If given multiple values will try all.')

args = parser.parse_args()

out_dir = '/'.join(args.out_file.split('/')[:-1])+'/'
out_fname = args.out_file.split('/')[-1]

#set up for parallelization
ncpus = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
pool = Pool(processes=ncpus)
print(f'Connected to pool with {ncpus} cpus.',flush=True)

#############
## load and filter all assembled sequences
#############

#read all files into big dataframe (this will have duplicates)
files = glob(f'{args.in_dir}*db-pass.tsv')
if len(files) == 0:
    print("Can't find tsv files - looking one dir deeper.")
    files = glob(f'{args.in_dir}*/*db-pass.tsv')
    
print('Reading the following files:')
print('\n'.join([f.split('/')[-1] for f in files]))
data = pd.concat([pd.read_csv(tsv,sep='\t') for tsv in tqdm(files)],axis=0)
#drop less useful info
data = data[["sequence", "locus", "junction", "junction_aa", "v_call", "d_call", "j_call", "duplicate_count"]]

print(f'{len(data)} rows in global set after loading tsvs.', flush=True)

#define functions for aggregation
agg_funcs = {}
for c in data.columns:
    #take the first of each VHH group for descriptor columns
    if c != 'duplicate_count':
        agg_funcs[c] = lambda x: x.iloc[0]
    #sum the duplicate count
    else:
        agg_funcs[c] = np.sum

#aggregate based on sequence
data = data.groupby(by='sequence',sort=False).aggregate(func=agg_funcs)
data.reset_index(inplace=True,drop=True) #not sure why, but need drop=True here to avoid error, despite not needing this in other cases
print(f'{len(data)} unique DNA sequences.', flush=True)

#get translations based on scanning for start and end sequences
seq_info = pd.DataFrame(pool.starmap(translate_VHH_end_scan,
                                     [(seq, args.start_seq, args.end_seq, args.max_start_scan, args.max_end_scan) for seq in data['sequence']],
                                     chunksize=5000),
                        columns=['sequence','VHH','start_match','start_i','start_f','end_match','end_i'])
data = data.merge(seq_info,on='sequence',how='outer')
print(f'{len(data)} unique DNA sequences after merging translation info.', flush=True)

#get filtering statistics
has_stop = data['VHH'].map(lambda x: '*' in x)
bad_start = data['start_match'] < args.min_start_match
bad_end = data['end_match'] < args.min_end_match
print(f'{np.sum(has_stop)} ({round(np.mean(has_stop)*100,2)}%) of sequences have an early stop.', flush=True)
print(f'{np.sum(bad_start)} ({round(np.mean(bad_start)*100,2)}%) of sequences have <{args.min_start_match} matches to the start sequence.',flush=True)
print(f'{np.sum(bad_end)} ({round(np.mean(bad_end)*100,2)}%) of sequences have <{args.min_start_match} matches to the end sequence.', flush=True)

#filter out bad sequences
data = data.loc[(~has_stop)&(~bad_start)&(~bad_end),:]
print(f'{len(data)} unique DNA sequences after filtering out these errors.', flush=True)

#get duplicate count of the AA seq encoded by the DNA sequence
vhh_agg_funcs = {'VHH':lambda x: x.iloc[0], 'duplicate_count':np.sum}
vhh_agg = data[['VHH','duplicate_count']].groupby('VHH').aggregate(func=vhh_agg_funcs)
vhh_agg.reset_index(inplace=True,drop=True) #not sure why, but need drop=True here to avoid error, despite not needing this in other cases
print(f'{len(vhh_agg)} unique VHHs.', flush=True)
vhh_agg = vhh_agg.rename({'duplicate_count':'VHH_duplicate_count'},axis=1)

#merge vhh duplicate counts onto main df
data = data.merge(vhh_agg,on='VHH',how='outer')
print(f'{len(data)} rows after merging with AA-level data.', flush=True)

#drop sequences who's translations are observed only once across the whole dataset
data = data.loc[data['VHH_duplicate_count']>1,:]
print(f'{len(data)} unique DNA sequnces, encoding {len(data["VHH"].unique())} unique VHHs after dropping singleton AA sequences.', flush=True)
data.index = range(len(data)) #reset index

#make sequence_id column
data['sequence_id'] = data.index.map(lambda i: 'GS-'+str(i))

#drop duplicate full-length amino acid seqs (no need to track counts b/c we already did this)
data = data.sort_values(by='duplicate_count',ascending=False) #put in order of highest counts
vhh = data.drop_duplicates('VHH',keep='first') #keep most abundant
print(f'{len(vhh)} unique VHHs.', flush=True)

#############
## get parameter combos
#############

recursive_min_id_vals = np.linspace(*args.recursive_min_id_start,args.recursive_min_id_start_N)

cdr3_dist_combos = itertools.product(np.linspace(*args.CDR3_weight,args.CDR3_weight_N),
                                     np.linspace(*args.max_len_diff,args.max_len_diff_N),
                                     np.linspace(*args.len_diff_penalty,args.len_diff_penalty_N))
cdr3_dist_combos = list(cdr3_dist_combos)
                                     
weighted_clustering_combos = itertools.product(args.linkage,
                                               np.linspace(*args.weighted_min_id,args.weighted_min_id_N))
weighted_clustering_combos = list(weighted_clustering_combos)

#############
## group sequences based on V and J identity
#############

tmp_dir = f'{out_dir}tmp_{out_fname.replace(".csv","")}'
os.mkdir(tmp_dir)

print(f'Getting SCOPer groups.', flush=True)

#get basic V/J clustering from Immcantation (scoper::hierarchicalClones)
#make fake column to spoof CDR length/clustering
vhh['fake_junction'] = 'A'
#run hclust in R
vhh.to_csv(f'{out_dir}tmp_input_{out_fname}')
code = subprocess.run(['Rscript', '/cluster/tufts/cowenlab/wwhite06/vhh/scripts/hclust_V-J-len_only_util.R',
                      f'{out_dir}tmp_input_{out_fname}', f'{out_dir}tmp_output_{out_fname}', 'fake_junction']).returncode
assert code==0
#get hclust data
vhh = pd.read_csv(f'{out_dir}tmp_output_{out_fname}')
#clean up tmp files
os.remove(f'{out_dir}tmp_input_{out_fname}')
os.remove(f'{out_dir}tmp_output_{out_fname}')

#delete fake column
del vhh['fake_junction']

#get ANARCI annotations
print(f'Getting ANARCI parses.', flush=True)
anarci_info = pd.DataFrame(pool.map(get_anarci_alignment, vhh['VHH'],chunksize=1000),
                          columns=['VHH','anarci_parse','CDR1','CDR2','CDR3','ANARCI_success'])
vhh = vhh.merge(anarci_info[['VHH','anarci_parse', 'CDR1','CDR2','CDR3','ANARCI_success']],on='VHH',how='outer')
print(f'{len(vhh)} rows after merging with ANARCI data.', flush=True)

#############
## cluster sequences based full length AA sequence and weighting scheme
#       repeat this for all possible parameter combos given by user
#############

#cluster within SCOPer groups
min_id = defaultdict(int)
for g,scoper_group in vhh.groupby(by='scoper_group'):
    
    #try all recursive_min_id values for this SCOPer group
    for rec_idx,rec_min_id in enumerate(recursive_min_id_vals):
    
        #recursively split into smaller subgroups with mmseqs
        scoper_group, max_min_id = recursive_mmseqs_split(scoper_group, args.max_subgroup_size, rec_min_id, args.recursive_min_id_step, tmp_dir, ncpus)
        
        #cluster by weighted CDR distance within each broad mmseqs cluster
        for mm_id,mmseqs_group in tqdm(scoper_group.groupby('mmseqs_id')):
            
            #try all CDR3 weight values
            for cdr_idx,dist_params in enumerate(cdr3_dist_combos):
            
                #skip this group if it has only one member
                if len(mmseqs_group) == 1:
                    
                    #save ID number for all clustering param combos
                    for clust_idx,_ in enumerate(weighted_clustering_combos):
                        vhh.loc[mmseqs_group.index,f'clone_id_{rec_idx}_{cdr_idx}_{clust_idx}'] = min_id[(rec_idx,cdr_idx,clust_idx)] #use min_id as usual

                #actually do clustering for groups with >1 member
                else:
                    #get dist matrix
                    CDR_list_list = [row[['CDR1','CDR2','CDR3']].tolist() for _,row in mmseqs_group.iterrows()]
                    
                    dists = np.array(pool.starmap(ANARCI_dist_with_penalty_row,((CDR_list1,
                                                                                CDR_list_list,
                                                                                [1,1,dist_params[0]],
                                                                                [dist_params[1]]*3,
                                                                                dist_params[2]) for CDR_list1 in CDR_list_list),
                                                  chunksize=10))
                    #try all clustering param combos
                    for clust_idx,clust_params in enumerate(weighted_clustering_combos):
                        clusterer = AgglomerativeClustering(metric='precomputed',
                                                            linkage=clust_params[0],
                                                            distance_threshold=1-clust_params[1],
                                                            n_clusters=None)
                        labels = clusterer.fit_predict(dists) + min_id[(rec_idx,cdr_idx,clust_idx)]
                        # print(f'SCOPer: {g}, ({rec_idx},{cdr_idx},{clust_idx}): {min_id[(rec_idx,cdr_idx,clust_idx)]}, NaNs: {np.mean(np.isnan(labels))}',flush=True)
                        
                        #merge labels onto group info
                        vhh.loc[mmseqs_group.index,f'clone_id_{rec_idx}_{cdr_idx}_{clust_idx}'] = labels
            
                #update min_id for all clustering variations
                for clust_idx,_ in enumerate(weighted_clustering_combos):
                    min_id[(rec_idx,cdr_idx,clust_idx)] = vhh[f'clone_id_{rec_idx}_{cdr_idx}_{clust_idx}'].max() + 1
             
            vhh = vhh.copy() #do this to defragment after many column insertions
            
        #delete mmseqs_id column so it can be re-used in the next mmseqs recursive split
        del scoper_group['mmseqs_id']
            
print(f'Finished individual clusterings.', flush=True)

#defragment by copying df
vhh = vhh.copy()
      
#############
## filter individual clusterings
#############

id_cols = [col for col in vhh.columns if 'clone_id' in col]
#skip filtering if filter is set to 0
if args.score_fraction < 1:
    
    scores_df = []
    for col in tqdm(id_cols):
        col_scores = []
        for g,group in vhh.groupby(col):
            col_scores.append(cluster_ratio_score(group['CDR3'].unique(),args.dist_cutoff,pool))
        s = np.mean(col_scores)
        scores_df.append([col,s])
        print(f'Score: {col}, {s}',flush=True)
        
    scores_df = pd.DataFrame(scores_df,columns=['column','score'])
    scores_df = scores_df.sort_values(by='score',ascending=False)
    N_keep = int(len(scores_df)*args.score_fraction)
    scores_df = scores_df.head(N_keep)
    keep_cols = scores_df['column'].tolist()
    print(f'{len(keep_cols)} of {len(id_cols)} clusterings pass the score cutoff ({scores_df["score"].values[-1]}).',flush=True)
else:
    keep_cols = id_cols
    print(f'All clusterings pass the score cutoff.',flush=True)
      
#############
## meta clustering
#############

#group all VHHs that are always clustered together
for i,(g,group) in enumerate(vhh.groupby(by=keep_cols)):
    vhh.loc[group.index,'fine_clone_id'] = i
print(f'{len(vhh["fine_clone_id"].unique())} tight clusters found.',flush=True)

#run final coarse clustering with user-specified parameters

#get distances between fine clusters
#only need to compare one from each group because all within group are the same
group_reps = vhh.drop_duplicates('fine_clone_id')
#run clustering
for method in args.meta_method:
    links = linkage(group_reps[keep_cols].values,method=method,metric='hamming')
    for min_id in tqdm(args.meta_min_id):
        labels = fcluster(links,criterion='distance',t=1-min_id)
        group_reps.loc[:,f'meta_clone_id_{method}_{min_id}'] = labels
        
        print(f'{len(group_reps[f"meta_clone_id_{method}_{min_id}"].unique())} families found using {method} linkage at a {round(min_id*100,1)}% identity cutoff.',flush=True)

meta_id_cols = [col for col in group_reps.columns if 'meta_clone_id' in col]

#add results to vhh dataframe
print(f'{len(vhh)} VHHs before merge.',flush=True)
vhh = vhh.merge(group_reps[['fine_clone_id']+meta_id_cols],on='fine_clone_id',how='outer')
print(f'{len(vhh)} VHHs after merge.',flush=True)

#add results to main dataframe
print(f'{len(data)} sequences before merge.',flush=True)
if args.keep_intermediates:
    data = data.merge(vhh[['VHH','CDR1','CDR2','CDR3']+id_cols+meta_id_cols],on='VHH',how='outer')
else:
    data = data.merge(vhh[['VHH','CDR1','CDR2','CDR3']+meta_id_cols],on='VHH',how='outer')
print(f'{len(data)} sequences after merge.',flush=True)

#save results
data.to_csv(args.out_file)

#close parallel pool
pool.close()

#clean up
subprocess.run(['rm', '-r', tmp_dir])
