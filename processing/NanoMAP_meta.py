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
import networkx as nx
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
    parser.add_argument('--keep_intermediates',action='store_true',help='Set this flag to save the cluster IDs for each clustering leading to the metaclustering.')
    
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

    parser.add_argument('--single_dist_linspace',default=[0.05,0.4,10],nargs=3,type=float,help='Args passed to numpy linspace to generate maximum diatance cutoff values to use in single-linkage clustering step.')
    parser.add_argument('--average_dist_linspace',default=[0.05,0.8,10],nargs=3,type=float,help='Args passed to numpy linspace to generate maximum diatance cutoff values to use in average-linkage clustering step.')
    parser.add_argument('--complete_dist_linspace',default=[0.05,0.95,10],nargs=3,type=float,help='Args passed to numpy linspace to generate maximum diatance cutoff values to use in complete-linkage clustering step.')
    
    parser.add_argument('--merge_dist_linspace',type=float,nargs=3,default=[0.1,0.25,4],help='Args passed to numpy linspace to generate maximum diatance cutoff values to use in the merge step.')

    parser.add_argument('--dist_batch_size',type=int,default=1000,help='When calculating all pairwise distances between sequences in a set, break the calculation into sets of dist_batch_size (so that dist_batch_size**2 distances are calculated per batch).')
    
    #scoring args
    parser.add_argument('--keep_quants',type=float,nargs=2,default=[0.2,0.8],help='Keep the individual clusterings with scores between these quantiles. Only these clusterings are used in metaclustering.')
    parser.add_argument('--sample_N_pairs',type=int,default=10000,help='When calculating intra-family distances, calcualte weighted distances for a random sampling of sample_N_pairs pairs of sequences within the cluster.')
    
    #metaclusterign args
    parser.add_argument('--meta_max_dist',type=float,default=[0.75],nargs='*',help='Maximum fraction of disagreeing clusterings to group two VHHs. If given multiple values will try all.')
    parser.add_argument('--meta_method',type=str,default=['average'],nargs='*',help='Linkage method to use for meta clustering (can be single, average, or complete). If given multiple values will try all.')
    
    return parser.parse_args()
            
def get_greedy_rep_clustering(input_tuple):
    os.environ["NX_CUGRAPH_AUTOCONFIG"] = "True"
    
    #unpack inputs
    group, graph, id_col, cutoffs = input_tuple
    
    group['iloc_idxs'] = range(len(group))
    group_reps = group.drop_duplicates(id_col,keep='first').copy()

    subgraph = graph.subgraph(group_reps['iloc_idxs'].tolist()).copy()  #make a copy so the original is unaffeccted
    
    for dist_idx, max_dist in enumerate(cutoffs):
        dist_subgraph = subgraph.copy() #make a copy so the original is unaffeccted
        #remove edges with too high distance
        dist_subgraph.remove_edges_from([(u,v) for u, v, d in subgraph.edges(data=True) if d['dist']>max_dist])
        
        #get clusters
        graph_idxs = [node for node in dist_subgraph.nodes()] #assume this order matches the order igraph generates
        dist_subgraph = igraph.Graph.from_networkx(dist_subgraph)
        labels = leidenalg.find_partition(dist_subgraph, partition_type=leidenalg.ModularityVertexPartition, n_iterations=-1).membership
        
        #add labels to group
        group[f'{id_col}_{dist_idx}'] = np.nan
        group.iloc[graph_idxs,group.columns.get_loc(f'{id_col}_{dist_idx}')] = labels
        
        del dist_subgraph #save space
        
    return id_col, group

def scoper_metacluster(args,vhh,out_dir,out_fname,pool,ncpus):
    
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
    
    cut_vals = {
        'single':np.linspace(*args.single_dist_linspace[:2],int(args.single_dist_linspace[2])),
        'average':np.linspace(*args.average_dist_linspace[:2],int(args.average_dist_linspace[2])),
        'complete':np.linspace(*args.complete_dist_linspace[:2],int(args.complete_dist_linspace[2])),
    }
    methods = ['single','average','complete']
    
    #pre-allocate the new columns to avoid excessive de-fragmenting
    for method_idx, method in enumerate(methods):
        for cut_idx, cut in enumerate(cut_vals[method]):
            vhh[f'clone_id_{method_idx}_{cut_idx}'] = np.nan
        #defragment the dataframe
        vhh = vhh.copy()
        
    id_cols = [col for col in vhh.columns if 'clone_id' in col]
    
    #cluster within each alakazam group
    print('Clustering within Alakazam groups.',flush=True)
    next_label = defaultdict(int)
    for g,group in tqdm(vhh.groupby('vj_group',sort=False)):
        
        #drop out duplicated CDRs to save time/memory
        dedup = group.drop_duplicates(['CDR1','CDR2','CDR3'])
        print(f'Clustering group {g} containing {len(group)} VHHs with {len(dedup)} unique CDR combos.',flush=True)
        
        #skip groups with only one unique CDR combo
        if len(dedup)==1:
            for method_idx, method in enumerate(methods):
                for cut_idx, cut in enumerate(cut_vals[method]):
                    vhh.loc[group.index,f'clone_id_{method_idx}_{cut_idx}'] = next_label[(method_idx,cut_idx)]
        else:
            #get dist matrix
            CDR_list = [[np.array(list(row[cdr]), dtype='<U1') for cdr in ['CDR1','CDR2','CDR3']] for _,row in dedup.iterrows()]
            dists = squareform(batched_pairwise_weighted_anarci_dist_no_align(CDR_list, args.dist_batch_size, [1,1,args.CDR3_weight], pool))
            
            for method_idx, method in enumerate(methods):
                links = linkage(dists,method=method,optimal_ordering=False)
                for cut_idx, cut in enumerate(cut_vals[method]):
                    labels = fcluster(links,criterion='distance',t=cut) + next_label[(method_idx,cut_idx)]
                    dedup.loc[:,f'clone_id_{method_idx}_{cut_idx}'] = labels
            
            #merge deduplicated info back onto full group (have to temporarily move indexes into a column to merge doesn't mess them up)
            group = group[['CDR1','CDR2','CDR3']].reset_index().merge(dedup, on=['CDR1','CDR2','CDR3'], how='left').set_index('index')
            #insert new label info into vhh dataframe
            vhh.loc[group.index,id_cols] = group[id_cols]
                    
            del dists #save some memory
        
        #update next label
        for method_idx, method in enumerate(methods):
            for cut_idx, cut in enumerate(cut_vals[method]):
                next_label[(method_idx,cut_idx)] = vhh[f'clone_id_{method_idx}_{cut_idx}'].max() + 1
    
    #sort vhhs by duplicate count so drop duplicates will keep the most abundant ones
    vhh = vhh.sort_values(by='VHH_duplicate_count',ascending=False)
    
    #get ID column names
    id_cols = [col for col in vhh.columns if 'clone_id' in col]
    
    #get representatives of all groups
    rep_seqs_master = pd.DataFrame(columns=vhh.columns)
    for id_col in id_cols:
        rep_seqs_master = pd.concat([rep_seqs_master,vhh.drop_duplicates(id_col,keep='first')]).drop_duplicates('sequence_id')
    print(f'Found {len(rep_seqs_master)} total representatives across all clusterings.',flush=True)

    #split into rough groups with mmseqs
    print('Grouping representatives with MMseqs2.',flush=True)
    rep_seqs_master = run_mmseqs_clust(rep_seqs_master,
                                       1-args.merge_dist_linspace[1], #use the highest distance cutoff value, to be safe (full seq id used here should be higher than CDR dist anyway)
                                       'conncomp', tmp_dir, ncpus)
    
    #make sure order and indexing are consistent
    rep_seqs_master = rep_seqs_master.sort_values(by='VHH_duplicate_count',ascending=False)
    rep_seqs_master.reset_index(inplace=True,drop=True)
    
    print('Clustering representatives within MMseqs groups.',flush=True)
    
    #pre-make the merged ID columns to avoid excessive de-fragmenting
    merge_cuts = np.linspace(*args.merge_dist_linspace[:2],int(args.merge_dist_linspace[2]))
    for id_col in id_cols:
        for dist_idx in range(len(merge_cuts)):
            rep_seqs_master[f'{id_col}_{dist_idx}'] = np.nan
        rep_seqs_master = rep_seqs_master.copy() #defragment
    
    #perform merges within each mmseqs group
    next_label = defaultdict(int)
    for g, group in tqdm(rep_seqs_master.groupby('mmseqs_id',sort=False)):
        
        #don't bother clustering the singletons
        if len(group) == 1:
            for id_col in id_cols:
                for dist_idx in range(len(merge_cuts)):
                    rep_seqs_master.loc[group.index,f'{id_col}_{dist_idx}'] = next_label[f'{id_col}_{dist_idx}'] #add label
                    next_label[f'{id_col}_{dist_idx}'] += 1 #update next_label
        
        #cluster non-singletons
        else:
            #prep distance calculation batches
            CDR_list = [row[['CDR1','CDR2','CDR3']].tolist() for _,row in group.iterrows()]
            N_reps = len(CDR_list)
            batches = pairwise_batch_generator(CDR_list,args.dist_batch_size,[args.merge_dist_linspace[1], args.CDR3_weight, args.max_len_diff])
            
            #prep graph
            graph = nx.Graph()
            graph.add_nodes_from(range(N_reps))
            #node numbers are sequential indexes into group (iloc, not loc)
            
            #add edges
            for result in pool.imap_unordered(weighted_anarci_dist_all_pairs_filtered,batches):
                graph.add_edges_from(result) #indexes will correspond to indexes in rep_seqs_master
                
            #prepare parallel tasks
            par_inputs = ((group, graph, id_col, merge_cuts) for id_col in id_cols)
                
            for id_col, new_group in pool.imap_unordered(get_greedy_rep_clustering, par_inputs):
                
                #save group labels in master
                new_cols = [f'{id_col}_{dist_idx}' for dist_idx in range(len(merge_cuts))]
                rep_seqs_master.loc[new_group.index,new_cols] = new_group[new_cols] + [next_label[col] for col in new_cols]
                
                #update next_label
                for dist_idx in range(len(merge_cuts)):
                    next_label[f'{id_col}_{dist_idx}'] = rep_seqs_master[f'{id_col}_{dist_idx}'].max()+1
            
            del graph #save space
            
    print('Merging clusters based on representative clustering.',flush=True)
    #merge label info onto vhh df
    for id_col in id_cols:
        rep_seqs = rep_seqs_master.drop_duplicates(id_col,keep='first')[[id_col]+[f'{id_col}_{dist_idx}' for dist_idx in range(len(merge_cuts))]]
        vhh = vhh.merge(rep_seqs,on=id_col,how='left')
    
    #update to new ID columns (this includes the un-merged ones)
    id_cols = [col for col in vhh.columns if 'clone_id' in col]
    
    #skip filtering if all columns will be kept
    if (args.keep_quants[0] > 0) or (args.keep_quants[1] < 1):
        print('Scoring merged clusterings.',flush=True)
        dist_param_map = {col:(args.CDR3_weight,args.max_len_diff) for col in id_cols}
        score_df = weighted_ANARCI_silhouette_fast(vhh, id_cols, dist_param_map, args.sample_N_pairs, pool, ncpus)

        score_df = pd.DataFrame(score_df,columns=['name','score'])
        low = np.quantile(score_df['score'],args.keep_quants[0])
        high = np.quantile(score_df['score'],args.keep_quants[1])
        score_df = score_df[(score_df['score']>low)&(score_df['score']<high)]
        keep_cols = score_df['name'].tolist()
                
        print(f'{len(keep_cols)} of {len(id_cols)} clusterings have scores between {low} and {high}.',flush=True)
    else:
        keep_cols = id_cols
        print(f'Scoring step skipped, using all clusterings.',flush=True)
        
    #############
    ## meta clustering
    #############
    
    print('Starting meta-clustering.',flush=True)
    
    #group all VHHs that are always clustered together
    for i,(g,group) in enumerate(vhh.groupby(by=keep_cols,sort=False)):
        vhh.loc[group.index,'fine_clone_id'] = i
    print(f'{len(vhh["fine_clone_id"].unique())} tight clusters found.',flush=True)
    
    #run final coarse clustering with user-specified parameters
    
    #get distances between fine clusters
    #only need to compare one from each group because all within group are the same
    group_reps = vhh.drop_duplicates('fine_clone_id').copy()
    #run clustering
    dists = pdist(group_reps[keep_cols].to_numpy(copy=True),metric='hamming')
    for method in args.meta_method:
        links = linkage(dists,method=method)
        for max_dist in tqdm(args.meta_max_dist):
            labels = fcluster(links,criterion='distance',t=max_dist)
            group_reps.loc[:,f'meta_clone_id_{method}_{max_dist}'] = labels
            
            print(f'{len(group_reps[f"meta_clone_id_{method}_{max_dist}"].unique())} families found using {method} linkage at a {round(max_dist*100,1)}% distance cutoff.',flush=True)
        del links #save space
        
    meta_id_cols = [col for col in group_reps.columns if 'meta_clone_id' in col]
    
    #add results to vhh dataframe
    print(f'{len(vhh)} VHHs before merge.',flush=True)
    vhh = vhh.merge(group_reps[['fine_clone_id']+meta_id_cols],on='fine_clone_id',how='left')
    print(f'{len(vhh)} VHHs after merge.',flush=True)
    
    vhh = vhh.astype(dtype={col:int for col in id_cols+meta_id_cols})
    
    #clean up
    subprocess.run(['rm', '-r', tmp_dir])
        
    return vhh, id_cols, meta_id_cols
        
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
    vhh,id_cols,meta_id_cols = scoper_metacluster(args,vhh,out_dir,out_fname,pool,ncpus) #get cluster labels
    
    #add clustering results to main dataframe
    print(f'{len(data)} sequences before merge.',flush=True)
    if args.keep_intermediates:
        data = data.merge(vhh[['VHH','CDR1','CDR2','CDR3','vj_group']+id_cols+meta_id_cols],on='VHH',how='left')
    else:
        data = data.merge(vhh[['VHH','CDR1','CDR2','CDR3']+meta_id_cols],on='VHH',how='left')
    print(f'{len(data)} sequences after merge.',flush=True)
    data.to_csv(args.out_file)
    
    #close parallel pool
    pool.close()