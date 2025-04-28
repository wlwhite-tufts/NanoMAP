# Initial version of my new clustering method
# not directly used anymore - baked into the metaclustering script
# keep around because someone might want to use this without doing metaclustering


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

sys.path.append('/cluster/tufts/cowenlab/wwhite06/packages/VHH-clustering/src/utils/')
from python_utils import *

#get user inputs
parser = argparse.ArgumentParser(prog='Cluster Based on ANARCI alignment',
                                 description='Collect sequences from input folder and cluster them using an agglomerative clustering method.')
#I/O args
parser.add_argument('--in_dir',type=str,help='The directory that contains the assembled sequences. Looks for *db-pass.tsv, or */*db-pass.tsv')
parser.add_argument('--out_file',type=str,help='Name for the output file (can include file path).')
parser.add_argument('--out_columns',type=str,nargs='*',default=[],help='Which columns to keep in the output. Leave blank to keep all.')

#translation args
parser.add_argument('--start_seq',type=str,default='LVQPGGSLRLSC',help='Amino acid sequence to scan for when finding starting point of translation.')
parser.add_argument('--end_seq',type=str,default='WGQGTQVTVSS',help='Amino acid sequence to scan for when finding ending point of translation.')
parser.add_argument('--min_start_match',type=int,default=5,help='Minimum number of amino acids matching start_seq to be considered a real sequence.')
parser.add_argument('--min_end_match',type=int,default=5,help='Minimum number of amino acids matching end_seq to be considered a real sequence.')
parser.add_argument('--max_start_scan',type=int,default=40,help='Number of N-term amino acid positions to scan when looking for start_seq.')
parser.add_argument('--max_end_scan',type=int,default=40,help='Number of C-term amino acid positions to scan when looking for end_seq.')

#clustering args
parser.add_argument('--max_subgroup_size',type=int,default=25000,help='Maximum number of VHHs in a SCOPer (sub)group before triggering a recursive loose mmseqs sub-clustering.')
parser.add_argument('--recursive_min_id_start',type=float,default=0.5,help='Starting value for minimum sequence identity allowed for recursive loose mmseqs clustering.')
parser.add_argument('--recursive_min_id_step',type=float,default=0.025,help='Step size value for minimum sequence identity incerment in each recursive layer.')
parser.add_argument('--weight_scheme',type=float,nargs=3,default=[3,3,10], help='The weighting scheme for The CDRs. weight_scheme[i] represents the per-amino-acid weight of mismatches in CDR-i. ')
parser.add_argument('--max_len_diffs',type=int,nargs=3,default=[2,2,5], help='The maximum difference in length for CDRs used in weighted distance clustering.'+\
                                                                               'max_len_diffs[i] represents max diference allowed in CDR-i.')
parser.add_argument('--linkage',type=str,default='single',help='The linkage type to use in weighted hierarchical clustering step (can be single, complete, or average).')
parser.add_argument('--weighted_min_id',type=float,default=0.5,help='The distance cutoff to use in the weighted hierarchical clustering step.')
parser.add_argument('--narrow_mmseqs_min_id',type=float,default=0.99,help='Minimum sequence identity allowed for final narrow mmseqs clustering.')
                    
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

#############
## group sequences based on V and J identity, and CDR3 length
#############

print(f'Getting SCOPer groups.', flush=True)

#get basic V/J clustering from Immcantation (scoper::hierarchicalClones)
#make fake column to spoof CDR length/clustering
data['fake_junction'] = 'A'
#run hclust in R
data.to_csv(f'{out_dir}tmp_input_{out_fname}')
code = subprocess.run(['Rscript', '/cluster/tufts/cowenlab/wwhite06/vhh/scripts/hclust_V-J-len_only_util.R',
                      f'{out_dir}tmp_input_{out_fname}', f'{out_dir}tmp_output_{out_fname}', 'fake_junction']).returncode
assert code==0
#get hclust data
data = pd.read_csv(f'{out_dir}tmp_output_{out_fname}')
#clean up tmp files
os.remove(f'{out_dir}tmp_input_{out_fname}')
os.remove(f'{out_dir}tmp_output_{out_fname}')

#delete fake column
del data['fake_junction']

#drop duplicate full-length amino acid seqs (no need to track counts b/c we already did this)
data = data.sort_values(by='duplicate_count',ascending=False) #put in order of highest counts
vhh = data.drop_duplicates('VHH',keep='first') #keep most abundant
print(f'{len(vhh)} unique VHHs.', flush=True)

#############
## cluster sequences based full length AA sequence and weighting scheme
#############

#get ANARCI annotations
print(f'Getting ANARCI parses.', flush=True)
anarci_info = pd.DataFrame(pool.map(get_anarci_alignment, vhh['VHH'],chunksize=1000),
                          columns=['VHH','anarci_parse','CDR1','CDR2','CDR3','ANARCI_success'])
vhh = vhh.merge(anarci_info[['VHH','anarci_parse', 'CDR1','CDR2','CDR3','ANARCI_success']],on='VHH',how='outer')
print(f'{len(vhh)} rows after merging with ANARCI data.', flush=True)

#cluster within SCOPer groups
groups = []
min_id = 0
tmp_dir = f'{out_dir}tmp_{out_fname.replace(".csv","")}'
os.mkdir(tmp_dir)
for g,scoper_group in tqdm(vhh.groupby(by='scoper_group')):
    
    #recursively split into smaller subgroups with mmseqs
    scoper_group, max_min_id = recursive_mmseqs_split(scoper_group, args.max_subgroup_size, args.recursive_min_id_start, args.recursive_min_id_step, tmp_dir, ncpus)
    
    print(f'Clustered {len(scoper_group)} VHHs in SCOPer group {g} into {len(scoper_group["mmseqs_id"].unique())} clusters with highest min_id = {max_min_id}.', flush=True)
    
    #cluster by weighted CDR distance within each broad mmseqs cluster
    for mm_id,mmseqs_group in scoper_group.groupby('mmseqs_id'):
        
        #skip this group if it has only one member
        if len(mmseqs_group) == 1:
            mmseqs_group['weighted_id'] = min_id #use min_id as usual
        
        #actually do clustering for groups with >1 member
        else:
            #get dist matrix and cluster
            CDR_list_list = [row[['CDR1','CDR2','CDR3']].tolist() for _,row in mmseqs_group.iterrows()]
            dists = np.array(pool.starmap(ANARCI_dist_row,((CDR_list1,CDR_list_list,args.weight_scheme,args.max_len_diffs) for CDR_list1 in CDR_list_list),
                                          chunksize=10))
            
            clusterer = AgglomerativeClustering(metric='precomputed',
                                                linkage=args.linkage,
                                                distance_threshold=1-args.weighted_min_id,
                                                n_clusters=None)
            labels = clusterer.fit_predict(dists) + min_id
            
            #merge labels onto group info
            mmseqs_group['weighted_id'] = labels
    
        #update min_id and save group
        min_id = mmseqs_group['weighted_id'].max() + 1
        groups.append(mmseqs_group)
        
        print(f'\t Clustered {len(mmseqs_group)} VHHs in mmseqs group {mm_id} into {len(mmseqs_group["weighted_id"].unique())} clusters.', flush=True)
    
vhh = pd.concat(groups,axis=0)
print(f'Finished within-SCOPer clustering. Found {len(vhh["weighted_id"].unique())} clusters', flush=True)
    
#############
## final cluster merging
#############

#get all pairwise VHH distances with mmseqs
print('Calculating final clusters with mmseqs.',flush=True)

#get group representatives (most abundant member of each group)
rep_seqs = vhh.sort_values(by='VHH_duplicate_count',ascending=False)[['weighted_id','VHH']].groupby('weighted_id').first()

#write fasta file
with open(f'{tmp_dir}/reps.fasta','w') as f:
    for i,row in rep_seqs.iterrows():
        f.write(f">{i}\n{row['VHH']}\n") #Index here is the weighted_id
        
#run mmseqs clustering
result = subprocess.run(['/cluster/tufts/cowenlab/emosel01/condaenv/vhh/bin/mmseqs', 'easy-cluster', f'{tmp_dir}/reps.fasta', f'{tmp_dir}/clust', f'{tmp_dir}/tmp',
                         '-s', '7.5', #sensitivity set to max
                         '-c', str(max(0,(args.narrow_mmseqs_min_id - 0.2))), #set this lower than seq-id cutoff so it doesn't have an impact
                         '--min-seq-id', str(args.narrow_mmseqs_min_id), #only cluster seqs more similar than user input
                         '-e', 'inf', #max e-value to be clustered (set to keep all)
                         '-v', '1', #verbosity
                         '--cluster-steps', '3', #use faster cascaded clustering method
                         '--seq-id-mode', '0', #use alignment length to normalize
                         '--cluster-mode', '0', #use greedy setcover algorithm
                         '--alignment-mode', '3', #get percent identity, rather than alignment score
                         '--max-seqs', '1000', #get top 1000 matches to each query
                         '--gap-open', '20', #set high gap-opening penalty to prevent excessive gaps
                         '--gap-extend', '3',#set high gap-extend penalty to prevent excessive gaps
                         '--similarity-type', '2', #use sequence identity, not score
                         '--threads', str(ncpus)]) #how many cpus to use
assert result.returncode == 0

#read in clusters
final_clusters = pd.read_csv(f'{tmp_dir}/clust_cluster.tsv',names=['clone_id','weighted_id'],sep='\t')

#merge on final cluster info
print(f'Finished between-SCOPer clustering. Found {len(final_clusters["clone_id"].unique())} clusters.', flush=True)
print(f'Merging cluster info on. {len(vhh)} rows before clustering', flush=True)
vhh = vhh.merge(final_clusters,on='weighted_id',how='left')
print(f'{len(vhh)} rows after merging cluster info.', flush=True)

#merge vhh-level info back onto DNA-level info
print(f'DNA-level data has {len(data)} rows before merging VHH-level info.', flush=True)
data = data.merge(vhh[['VHH','CDR1','CDR2','CDR3','ANARCI_success','mmseqs_id','weighted_id','clone_id']],on='VHH',how='outer') #use only necessary columns
print(f'{len(data)} rows after merging.', flush=True)

#close parallel pool
pool.close()

#clean up
subprocess.run(['rm', '-r', tmp_dir])

#convert labels to integers
data = data.astype(dtype={'weighted_id':int,'clone_id':int})

#save requested columns to csv
if len(args.out_columns) == 0:
    data.to_csv(args.out_file)
else:
    data[args.out_columns].to_csv(args.out_file)