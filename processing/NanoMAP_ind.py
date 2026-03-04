import numpy as np
import pandas as pd
import sys
import os
import shutil
import subprocess
from multiprocessing import Pool
from glob import glob
from tqdm import tqdm
import argparse
import itertools
from collections import defaultdict
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform, pdist
from Levenshtein import distance
from functools import partial
import igraph
import leidenalg

#get main dir for repo
repo_path = os.path.dirname(os.path.abspath(__file__))
repo_path = '/'.join(repo_path.split('/')[:-1])
sys.path.append(repo_path)
from utils.utils import *
from analysis.fast_silhouette_score import weighted_ANARCI_silhouette_fast

def get_args():
    #get user inputs
    parser = argparse.ArgumentParser(prog='Cluster Based on ANARCI alignment',
                                     description='Collect sequences from input folder (or subfolders directly within it) and cluster them using an agglomerative meta-clustering method.')
    #I/O args
    parser.add_argument('--in_dir',type=str,help='The directory that contains the assembled sequences. Looks for *db-pass.tsv, or */*db-pass.tsv')
    parser.add_argument('--out_file',type=str,help='Name for the output file (can include file path).')

    #translation args
    parser.add_argument('--start_seq',type=str,default='LVQPGGSLRLSC',help='Amino acid sequence to scan for when finding starting point of translation.')
    parser.add_argument('--end_seq',type=str,default='WGQGTQVTVSS',help='Amino acid sequence to scan for when finding ending point of translation.')
    parser.add_argument('--min_start_match',type=int,default=5,help='Minimum number of amino acids matching start_seq to be considered a real sequence.')
    parser.add_argument('--min_end_match',type=int,default=5,help='Minimum number of amino acids matching end_seq to be considered a real sequence.')
    parser.add_argument('--max_start_scan',type=int,default=40,help='Number of N-term amino acid positions to scan when looking for start_seq.')
    parser.add_argument('--max_end_scan',type=int,default=40,help='Number of C-term amino acid positions to scan when looking for end_seq.')
    parser.add_argument('--min_count',type=int,default=2,help='The minimum number of counts for an amino acid sequence for it to be retained for clustering')
    
    #clustering args
    parser.add_argument('--CDR3_weight',type=float,default=3, help='The weight for CDR3 in weighted distance calculations. Uses a weight of 1 for CDR1 and CDR2.')
    parser.add_argument('--max_len_diff',type=int,default=3, help='Maximum allowed difference in CDR length in weighted distance calculations.')

    parser.add_argument('--linkage',default='average',type=str,help='The linkage method to use in the initial hierarchical clustering step.')
    parser.add_argument('--max_dist',default=0.467,type=float,help='The distance cutoff to use in the initial hierarchical clustering step.')
    parser.add_argument('--merge_dist',type=float,default=0.25,help='The distance cutoff to use in the cluster merging step.')

    parser.add_argument('--dist_batch_size',type=int,default=1000,help='When calculating all pairwise distances between sequences in a set, break the calculation into sets of dist_batch_size (so that dist_batch_size**2 distances are calculated per batch).')
    
    return parser.parse_args()

def cluster(args,vhh,out_dir,out_fname,pool,ncpus):
    
    tmp_dir = f'{out_dir}tmp_{out_fname.replace(".csv","")}'
    if os.path.exists(tmp_dir):
        #remove stuff from previous run if it exists
        shutil.rmtree(tmp_dir)
    #make the folder fresh
    os.mkdir(tmp_dir)
    
    print(f'Getting Alakazam groups.', flush=True)
    
    #get basic V/J/length grouping from Immcantation (alakazam::groupGenes)
    #get a unique label for each combo of CDRs
    vhh['CDR_len_combo'] = vhh[['CDR1','CDR2','CDR3']].apply(get_CDR_len_combo,axis=1)
    
    #run grouping in R
    vhh.to_csv(f'{out_dir}tmp_input_{out_fname}')
    code = subprocess.run(['Rscript', f'{repo_path}/utils/alakazam_grouping_util.R',
                                      f'{out_dir}tmp_input_{out_fname}', #input file
                                      f'{out_dir}tmp_output_{out_fname}', #output file
                                      'CDR_len_combo']).returncode #col name inuqie CDR length combo label
    assert code==0
    #get hclust data
    vhh = pd.read_csv(f'{out_dir}tmp_output_{out_fname}',na_filter=False)
    del vhh['CDR_len_combo']
    #clean up tmp files
    os.remove(f'{out_dir}tmp_input_{out_fname}')
    os.remove(f'{out_dir}tmp_output_{out_fname}')
    
    #cluster within each alakazam group
    print('Clustering within Alakazam groups.',flush=True)
    next_label = 0
    for g,group in tqdm(vhh.groupby('vj_group',sort=False)):
        
        #drop out duplicated CDRs to save time/memory
        dedup = group.drop_duplicates(['CDR1','CDR2','CDR3']).copy()
        print(f'Clustering group {g} containing {len(group)} VHHs with {len(dedup)} unique CDR combos.',flush=True)
        
        #skip groups with only one unique CDR combo
        if len(dedup)==1:
            vhh.loc[group.index,'initial_id'] = next_label
            
        else:
            #get dist matrix
            CDR_list = [[np.array(list(row[cdr]), dtype='<U1') for cdr in ['CDR1','CDR2','CDR3']] for _,row in dedup.iterrows()]
            dists = squareform(batched_pairwise_weighted_anarci_dist_no_align(CDR_list, args.dist_batch_size, [1,1,args.CDR3_weight], pool))
            
            links = linkage(dists,method=args.linkage,optimal_ordering=False)
            labels = fcluster(links,criterion='distance',t=args.max_dist) + next_label
            dedup.loc[:,'initial_id'] = labels
            
            #merge deduplicated info back onto full group (have to temporarily move indexes into a column to merge doesn't mess them up)
            group = group[['CDR1','CDR2','CDR3']].reset_index().merge(dedup, on=['CDR1','CDR2','CDR3'], how='left').set_index('index')
            #insert new label info into vhh dataframe
            vhh.loc[group.index,'initial_id'] = group['initial_id']
                    
            del dists #save some memory
        
        #update next label
        next_label = vhh['initial_id'].max() + 1
    
    #sort vhhs by duplicate count so drop duplicates will keep the most abundant ones
    vhh = vhh.sort_values(by='VHH_duplicate_count',ascending=False)
    
    #get representative form each cluster
    rep_seqs = vhh.drop_duplicates('initial_id',keep='first')
    print(f'Found {len(rep_seqs)} initial clusters.',flush=True)

    #split into rough groups with mmseqs
    print('Grouping representatives with MMseqs2.',flush=True)
    rep_seqs = run_mmseqs_clust(rep_seqs,
                                1-args.merge_dist, #use the same distance cutoff value, to be safe (full seq id used here should be higher than CDR dist anyway)
                                'conncomp', tmp_dir, ncpus)
    
    #make sure order and indexing are consistent
    rep_seqs = rep_seqs.sort_values(by='VHH_duplicate_count',ascending=False)
    rep_seqs.reset_index(inplace=True,drop=True)
    
    print('Clustering representatives within MMseqs groups.',flush=True)
    
    #perform merges within each mmseqs group
    next_label = 0
    for g, group in tqdm(rep_seqs.groupby('mmseqs_id',sort=False)):
        
        #don't bother clustering the singletons
        if len(group) == 1:
            rep_seqs.loc[group.index,'clone_id'] = next_label #add label
            next_label += 1 #update next_label
        
        #cluster non-singletons
        else:
            #prep distance calculation batches
            CDR_list = [row[['CDR1','CDR2','CDR3']].tolist() for _,row in group.iterrows()]
            N_reps = len(CDR_list)
            batches = pairwise_batch_generator(CDR_list,args.dist_batch_size,[args.merge_dist, args.CDR3_weight, args.max_len_diff])
            
            #prep graph
            graph = igraph.Graph()
            graph.add_vertices(N_reps)
            #node numbers are sequential indexes into group (iloc, not loc)
            
            #add edges
            for result in pool.imap_unordered(weighted_anarci_dist_all_pairs_filtered,batches):
                graph.add_edges((e[:2] for e in result)) #indexes will correspond to iloc indexes in group
                
            #get Leiden clustering
            labels = leidenalg.find_partition(graph, partition_type=leidenalg.ModularityVertexPartition, n_iterations=-1).membership
            #labels should be provided in iloc order
                
            #save group labels in master
            rep_seqs.loc[group.index,'clone_id'] = np.array(labels) + next_label
                
            #update next_label
            next_label = rep_seqs['clone_id'].max()+1
            
            del graph #save space
            
    print('Merging clusters based on representative clustering.',flush=True)
    #merge label info onto vhh df
    vhh = vhh.merge(rep_seqs[['initial_id','clone_id']],on='initial_id',how='left').astype(dtype={'clone_id':int})
    
    #clean up
    subprocess.run(['rm', '-r', tmp_dir])
        
    return vhh
        
if __name__ == '__main__':
    
    args = get_args()
    
    #set up for parallelization
    ncpus = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
    pool = Pool(processes=ncpus)
    print(f'Connected to pool with {ncpus} cpus.',flush=True)
    
    #parse output folder and file names
    args.out_file = os.path.realpath(args.out_file)
    out_dir = '/'.join(args.out_file.split('/')[:-1])+'/'
    print('out_dir = ',out_dir)
    out_fname = args.out_file.split('/')[-1]
    
    #read all files into big dataframe (this will have duplicates)
    files = glob(f'{args.in_dir}*db-pass.tsv')
    if len(files) == 0:
        print("Can't find tsv files - looking one dir deeper.")
        files = glob(f'{args.in_dir}*/*db-pass.tsv')
        
    data = collect_seqs(files) #get sequence info from user-specificed folder
        
    #define functions for aggregation
    agg_funcs = {}
    for c in data.columns:
        #take the first of each VHH group for descriptor columns
        if c != 'duplicate_count':
            agg_funcs[c] = lambda x: x.iloc[0]
        #sum the duplicate count
        else:
            agg_funcs[c] = np.sum
    
    data,vhh = annotate_and_filter_seqs(args,data,pool,ncpus,agg_funcs) #get translations, CDR anotations, and filter out bad/singleton sequences
    
    #remove unnecessary column to save space
    del vhh['anarci_parse']
    
    #cluster
    vhh = cluster(args,vhh,out_dir,out_fname,pool,ncpus) #get cluster labels
    
    #add clustering results to main dataframe
    print(f'{len(data)} sequences before merge.',flush=True)
    data = data.merge(vhh[['VHH','CDR1','CDR2','CDR3','clone_id']],on='VHH',how='left')
    print(f'{len(data)} sequences after merge.',flush=True)
    data.to_csv(args.out_file)
    
    #close parallel pool
    pool.close()