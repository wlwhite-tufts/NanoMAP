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
sys.path.append('/cluster/tufts/cowenlab/wwhite06/packages/VHH-clustering/src/clustering/')
from meta_ANARCI_clustering_alt_edit import collect_seqs, annotate_and_filter_seqs

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

parser.add_argument('--recursive_method',type=str,nargs='*',default=['setcover'],help='Method to use for clustering in the recursive mmseqs clustering (setcover or conncomp).')

parser.add_argument('--CDR3_weight',type=float,nargs=2,default=[3.3,3.3], help='Inclusive bounds for the weight for CDR3 in weighted distance calculations. Uses a weight of 1 for CDR1 and CDR2.')
parser.add_argument('--CDR3_weight_N',type=int,default=1,help='Number of values to test between the CDR3_weight bounds.')

parser.add_argument('--max_len_diff',type=int,nargs=2,default=[3,3], help='Inclusive bounds for the range of values of maximum allowed difference in CDR length.')
parser.add_argument('--max_len_diff_N',type=int,default=1,help='Number of values to test between the max_len_diff bounds.')

parser.add_argument('--linkage',type=str,default=['single','average','complete'],nargs='+',help='The linkage methods to try in weighted hierarchical clustering step (can be single, complete, or average).')

parser.add_argument('--weighted_min_id',type=float,nargs=2,default=[0.3,0.7],help='Inclusinve bounds for the range of distance cutoff values to use in the weighted hierarchical clustering step.')
parser.add_argument('--weighted_min_id_N',type=int,default=3,help='Number of values to test between the weighted_min_id bounds.')

parser.add_argument('--final_merge_method',type=str,default=['complete'],nargs='+',help='The linkage method(s) to use in the final CDR-based merge.')

parser.add_argument('--final_min_id',type=float,nargs=2,default=[0.75,0.95],help='Inclusinve bounds for the range of distance cutoff values to use in the final merge step.')
parser.add_argument('--final_min_id_N',type=int,default=3,help='Number of values to test between the final_min_id bounds.')

parser.add_argument('--score_fraction',type=float,default=0.5,help='Keep the top score_fraction of individual clusterings. Only these clusterings are used in metaclustering.')
parser.add_argument('--score_top_N',type=int,default=100,help='When calculating inter-family distances, calcualte weighted distances to the top score_top_N MMseqs hits.')

parser.add_argument('--meta_min_id',type=float,default=[0.35],nargs='*',help='Minimum fraction of agreeing clusterings to group two VHHs. If given multiple values will try all. Clustering distance threshhold = 1 - meta_min_id')
parser.add_argument('--meta_method',type=str,default=['single'],nargs='*',help='Linkage method to use for meta clustering (can be single, average, or complete). If given multiple values will try all.')

def meta_ANARCI_clustering_CDR_merge(args,vhh,out_dir,out_fname,pool,ncpus):
    #############
    ## get parameter combos
    #############
    recursive_min_id_vals = itertools.product(args.recursive_method,
                                              np.linspace(*args.recursive_min_id_start,args.recursive_min_id_start_N))
    recursive_min_id_vals = list(recursive_min_id_vals)
    
    cdr3_dist_combos = itertools.product(np.linspace(*args.CDR3_weight,args.CDR3_weight_N),
                                         np.linspace(*args.max_len_diff,args.max_len_diff_N))
    cdr3_dist_combos = list(cdr3_dist_combos)
                                         
    weighted_clustering_combos = itertools.product(args.linkage,
                                                   np.linspace(*args.weighted_min_id,args.weighted_min_id_N))
    weighted_clustering_combos = list(weighted_clustering_combos)
    
    final_merge_combos = itertools.product(args.final_merge_method,
                                           np.linspace(*args.final_min_id,args.final_min_id_N))
    final_merge_combos = list(final_merge_combos)

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
    vhh = pd.read_csv(f'{out_dir}tmp_output_{out_fname}',na_filter=False)
    #clean up tmp files
    os.remove(f'{out_dir}tmp_input_{out_fname}')
    os.remove(f'{out_dir}tmp_output_{out_fname}')
    
    #delete fake column
    del vhh['fake_junction']
    
    #############
    ## cluster sequences based full length AA sequence and weighting scheme
    #       repeat this for all possible parameter combos given by user
    #############
    
    #cluster within SCOPer groups
    min_id = defaultdict(int)
    for g,scoper_group in vhh.groupby(by='scoper_group'):
        
        #try all recursive_min_id values for this SCOPer group
        for rec_idx,rec_params in enumerate(recursive_min_id_vals):
            rec_min_id = rec_params[1]
            rec_method = rec_params[0]
        
            #recursively split into smaller subgroups with mmseqs
            scoper_group, max_min_id = recursive_mmseqs_split(scoper_group, args.max_subgroup_size, rec_min_id, args.recursive_min_id_step, rec_method, tmp_dir, ncpus)
            
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
                        
                        dists = np.array(pool.starmap(ANARCI_dist_row,((CDR_list1,CDR_list_list,[1,1,dist_params[0]],[dist_params[1]]*3) for CDR_list1 in CDR_list_list)))
                        #try all clustering param combos
                        for clust_idx,clust_params in enumerate(weighted_clustering_combos):
                            clusterer = AgglomerativeClustering(metric='precomputed',
                                                                linkage=clust_params[0],
                                                                distance_threshold=1-clust_params[1],
                                                                n_clusters=None)
                            labels = clusterer.fit_predict(dists) + min_id[(rec_idx,cdr_idx,clust_idx)]
                            # print(f'SCOPer: {g}, ({rec_idx},{cdr_idx},{clust_idx}): {min_id[(rec_idx,cdr_idx,clust_idx)]}, NaNs: {np.mean(np.isnan(labels))}',flush=True)
                            
                            #merge labels onto group info
                            col_name = f'clone_id_{rec_idx}_{cdr_idx}_{clust_idx}'
                            
                            # Initialize column if it doesn't exist
                            if col_name not in vhh.columns:
                                vhh[col_name] = np.nan
                            
                            # Map row and column positions for iloc
                            row_pos = vhh.index.get_indexer(mmseqs_group.index)
                            col_pos = vhh.columns.get_loc(col_name)
                            
                            # Assign using iloc for speed
                            vhh.iloc[row_pos, col_pos] = labels
                            # print(f'Test: {(vhh.loc[mmseqs_group.index,"sequence_id"]==mmseqs_group["sequence_id"]).sum()}, {len(mmseqs_group)}, {len(vhh.loc[mmseqs_group.index,:])}',flush=True)
                
                    #update min_id for all clustering variations
                    for clust_idx,_ in enumerate(weighted_clustering_combos):
                        min_id[(rec_idx,cdr_idx,clust_idx)] = vhh[f'clone_id_{rec_idx}_{cdr_idx}_{clust_idx}'].max() + 1
                
            #delete mmseqs_id column so it can be re-used in the next mmseqs recursive split
            del scoper_group['mmseqs_id']
                
    print(f'Finished primary individual clusterings, starting merges.', flush=True)
    
    #defragment by copying df
    vhh = vhh.copy()
    vhh.to_csv(f'{tmp_dir}/intermediate.csv')
    
    #############
    ## merge very similar clusters (using CDR distance)
    #############
    
    id_cols = [col for col in vhh.columns if 'clone_id' in col]
    
    #for each of the above clusterings ... try all requested merging values
    for col in tqdm(id_cols):
    
        #get group representatives (most abundant member of each group)
        rep_seqs = vhh.sort_values(by='VHH_duplicate_count',ascending=False)[[col,'CDR1','CDR2','CDR3']].groupby(col).first().reset_index()

        #calculate pairwise distances between group reps
        CDR_list_list = [row[['CDR1','CDR2','CDR3']].tolist() for _,row in rep_seqs.iterrows()]
        dists = np.array(pool.starmap(ANARCI_dist_row,((CDR_list1,CDR_list_list,[1,1,np.mean(args.CDR3_weight)],[np.mean(args.max_len_diff)]*3) for CDR_list1 in CDR_list_list)))
        
        #try each requested merge min_id and method
        for merge_idx,(merge_method, merge_min_id) in enumerate(final_merge_combos):
            
            #get labels
            clusterer = AgglomerativeClustering(metric='precomputed',
                                                linkage=merge_method,
                                                distance_threshold=1-merge_min_id,
                                                n_clusters=None)
            rep_seqs.loc[:,f'{col}_{merge_idx}'] = clusterer.fit_predict(dists)
        
        #merge on final cluster info
        vhh = vhh.merge(rep_seqs[[col]+[f'{col}_{merge_idx}' for merge_idx in range(len(final_merge_combos))]],on=col,how='left')
        
        #delete original column since it is no longer needed
        del vhh[col]
        
        #defragment by copying df (and reset index)
        vhh = vhh.copy()
    
    #save checkpoint
    vhh.to_csv(f'{tmp_dir}/intermediate.csv')
    
    print(f'Finished merges, starting quality filtering.', flush=True)
        
    #############
    ## filter individual clusterings
    #############
    
    id_cols = [col for col in vhh.columns if 'clone_id' in col]
    #skip filtering if all columns will be kept
    if args.score_fraction < 1:
        
        score_df = weighted_ANARCI_silhouette(vhh, id_cols, args.score_top_N, [1,1,np.mean(args.CDR3_weight)], [np.mean(args.max_len_diff)]*3, pool, tmp_dir+'_mmseqs')
        
        score_df = score_df.sort_values(by='score',ascending=False)
        score_df = score_df.head(int(len(score_df)*args.score_fraction))
        keep_cols = score_df['name'].tolist()
                
        print(f'{len(keep_cols)} of {len(id_cols)} clusterings pass the score cutoff ({score_df["score"].min()}).',flush=True)
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
    vhh = vhh.merge(group_reps[['fine_clone_id']+meta_id_cols],on='fine_clone_id',how='left')
    print(f'{len(vhh)} VHHs after merge.',flush=True)
    
    vhh = vhh.astype(dtype={col:int for col in id_cols+meta_id_cols})
    
    #clean up
    subprocess.run(['rm', '-r', tmp_dir])
    
    return vhh,id_cols,meta_id_cols

if __name__ == '__main__':
    args = parser.parse_args()
    
    #set up for parallelization
    ncpus = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
    pool = Pool(processes=ncpus)
    print(f'Connected to pool with {ncpus} cpus.',flush=True)
    
    #parse output folder anf file names
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
    vhh,id_cols,meta_id_cols = meta_ANARCI_clustering_CDR_merge(args,vhh,out_dir,out_fname,pool,ncpus) #get cluster labels
    
    #add clustering results to main dataframe
    print(f'{len(data)} sequences before merge.',flush=True)
    if args.keep_intermediates:
        data = data.merge(vhh[['VHH','CDR1','CDR2','CDR3','scoper_group']+id_cols+meta_id_cols],on='VHH',how='left')
    else:
        data = data.merge(vhh[['VHH','CDR1','CDR2','CDR3']+meta_id_cols],on='VHH',how='left')
    print(f'{len(data)} sequences after merge.',flush=True)
    data.to_csv(args.out_file)
    
    #close parallel pool
    pool.close()