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
from scipy.spatial.distance import squareform
from Levenshtein import distance
from functools import partial

sys.path.append('/cluster/tufts/cowenlab/wwhite06/packages/VHH-clustering/src/utils/')
from python_utils import *
sys.path.append('/cluster/tufts/cowenlab/wwhite06/packages/VHH-clustering/src')
from clustering.anarci.meta_ANARCI_clustering_alt_edit import collect_seqs, annotate_and_filter_seqs
from analysis.sequence_silhouette_score import weighted_ANARCI_silhouette

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
parser.add_argument('--recursive_min_id_start_N',type=int,default=5,help='Number of values to test between the recursive_min_id_start bounds.')
parser.add_argument('--recursive_min_id_step',type=float,default=0.025,help='Step size value for minimum sequence identity incerment in each recursive layer.')

parser.add_argument('--recursive_method',type=str,nargs='*',default=['setcover'],help='Method to use for clustering in the recursive mmseqs clustering (setcover or conncomp).')

parser.add_argument('--CDR3_weight',type=float,nargs=2,default=[3,3], help='Inclusive bounds for the weight for CDR3 in weighted distance calculations. Uses a weight of 1 for CDR1 and CDR2.')
parser.add_argument('--CDR3_weight_N',type=int,default=1,help='Number of values to test between the CDR3_weight bounds.')

parser.add_argument('--max_len_diff',type=int,nargs=2,default=[3,3], help='Inclusive bounds for the range of values of maximum allowed difference in CDR length.')
parser.add_argument('--max_len_diff_N',type=int,default=1,help='Number of values to test between the max_len_diff bounds.')

parser.add_argument('--linkage',type=str,default=['single','average','complete'],nargs='+',help='The linkage methods to try in weighted hierarchical clustering step (can be single, complete, or average).')

parser.add_argument('--weighted_min_id',type=float,nargs=2,default=[0.4,0.8],help='Inclusinve bounds for the range of distance cutoff values to use in the weighted hierarchical clustering step.')
parser.add_argument('--weighted_min_id_N',type=int,default=5,help='Number of values to test between the weighted_min_id bounds.')

parser.add_argument('--final_merge_method',type=str,default=['complete'],nargs='+',help='The linkage method(s) to use in the final CDR-based merge.')

parser.add_argument('--final_min_id',type=float,nargs=2,default=[0.75,0.95],help='Inclusinve bounds for the range of distance cutoff values to use in the final merge step.')
parser.add_argument('--final_min_id_N',type=int,default=3,help='Number of values to test between the final_min_id bounds.')

#scoring args
parser.add_argument('--score_fraction',type=float,default=0.5,help='Keep the top score_fraction of individual clusterings. Only these clusterings are used in metaclustering.')
parser.add_argument('--score_top_N',type=int,default=10,help='When calculating inter-family distances, calcualte weighted distances to the top score_top_N MMseqs hits for each family representative.')
parser.add_argument('--min_clust_reps',type=int,default=10,help='When calculating inter-family distances, select at least the top min_clust_reps most abundant members to represent each family.')
parser.add_argument('--max_extra_reps',type=int,default=150,help='When calculating inter-family distances, split clusterings into groups of no more than max_group_size members based on the overlap of their representatives.')
parser.add_argument('--sample_N_pairs',type=int,default=1000,help='When calculating intra-family distances, calcualte weighted distances for a random sampling of sample_N_pairs pairs of sequences within the cluster.')
# parser.add_argument('--min_id_for_approx_dist',type=float,default=0.9,help='When updating intra-family distances during merge step, use this fraction identity cutoff in mmseqs clusteing to find cluster representatives.')
# parser.add_argument('--min_cluster_size_shortcut',type=int,default=300,help='When updating intra-family distances during merge step, shortcut the calculation for each merge where both clusters are larger than min_cluster_size_shortcut.')

#metaclusterign args
parser.add_argument('--meta_min_id',type=float,default=[0.55],nargs='*',help='Minimum fraction of agreeing clusterings to group two VHHs. If given multiple values will try all. Clustering distance threshhold = 1 - meta_min_id')
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
                    
                    # Map row positions for iloc
                    row_pos = vhh.index.get_indexer(mmseqs_group.index)
                
                    #skip this group if it has only one member
                    if len(mmseqs_group) == 1:
                        
                        #save ID number for all clustering param combos
                        for clust_idx,_ in enumerate(weighted_clustering_combos):
                            
                            #save ID number for all clustering param combos
                            clust_col = f'clone_id_{rec_idx}_{cdr_idx}_{clust_idx}'
                            # dist_col = f'farthest_{rec_idx}_{cdr_idx}_{clust_idx}'
                            
                            # Initialize column if it doesn't exist
                            if clust_col not in vhh.columns:
                                vhh[clust_col] = np.nan
                                # vhh[dist_col] = np.nan
                            
                            #record cluster label and dist to furthest within cluster
                            vhh.iloc[row_pos, vhh.columns.get_loc(clust_col)] = min_id[(rec_idx,cdr_idx,clust_idx)] #use min_id as usual
                            # vhh.iloc[row_pos, vhh.columns.get_loc(dist_col)] = 0
    
                    #actually do clustering for groups with >1 member
                    else:
                        #get dist matrix
                        CDR_list = [row[['CDR1','CDR2','CDR3']].tolist() for _,row in mmseqs_group.iterrows()]
                        dists = np.array(pool.map(partial(ANARCI_dist_row,CDR_list_list=CDR_list,weight_scheme=[1,1,dist_params[0]],max_len_diffs=[dist_params[1]]*3),CDR_list))

                        #try all clustering param combos
                        for clust_idx,clust_params in enumerate(weighted_clustering_combos):
                            clusterer = AgglomerativeClustering(metric='precomputed',
                                                                linkage=clust_params[0],
                                                                distance_threshold=1-clust_params[1],
                                                                n_clusters=None)
                            labels = clusterer.fit_predict(dists) + min_id[(rec_idx,cdr_idx,clust_idx)]

                            #merge labels onto group info
                            clust_col = f'clone_id_{rec_idx}_{cdr_idx}_{clust_idx}'
                            # dist_col = f'farthest_{rec_idx}_{cdr_idx}_{clust_idx}'
                            
                            # Initialize column if it doesn't exist
                            if clust_col not in vhh.columns:
                                vhh[clust_col] = np.nan
                                # vhh[dist_col] = np.nan
                            
                            # Assign using iloc for speed
                            vhh.iloc[row_pos, vhh.columns.get_loc(clust_col)] = labels
                            
                            #get dist to furthest within cluster for each cluster without expanding condensed distance matrix
                            # furthest_dists = np.zeros([len(labels)])*np.nan
                            # for l in np.unique(labels):
                            #     idxs = labels==l
                            #     furthest_dists[idxs] = np.max(dists[idxs,:][:,idxs],axis=0)
                            # vhh.iloc[row_pos, vhh.columns.get_loc(dist_col)] = furthest_dists
                    
                    #update min_id for all clustering variations
                    for clust_idx,_ in enumerate(weighted_clustering_combos):
                        min_id[(rec_idx,cdr_idx,clust_idx)] = vhh[f'clone_id_{rec_idx}_{cdr_idx}_{clust_idx}'].max() + 1
                
            #delete mmseqs_id column so it can be re-used in the next mmseqs recursive split
            del scoper_group['mmseqs_id']
            
            #defragment the dataframe
            vhh = vhh.copy()
                
    print(f'Finished primary individual clusterings, starting merges.', flush=True)
    
    vhh.to_csv(f'{tmp_dir}/intermediate.csv')
    
    #############
    ## merge very similar clusters (using CDR distance)
    #############
    
    id_cols = [col for col in vhh.columns if 'clone_id' in col]
    
    #for each of the above clusterings ... try all requested merging values
    for id_col in tqdm(id_cols):
        
        # dist_col = id_col.replace('clone_id','farthest')
        
        #get group representatives (most abundant member of each group)
        rep_seqs = vhh.sort_values(by='VHH_duplicate_count',ascending=False)[[id_col,'CDR1','CDR2','CDR3']].groupby(id_col).first().reset_index()
        
        #calculate pairwise distances between group reps
        CDR_list = [row[['CDR1','CDR2','CDR3']].tolist() for _,row in rep_seqs.iterrows()]
        dists = np.array(pool.map(partial(ANARCI_dist_row,CDR_list_list=CDR_list,weight_scheme=[1,1,np.mean(args.CDR3_weight)],max_len_diffs=[np.mean(args.max_len_diff)]*3),CDR_list))

        #try each requested merge min_id and method
        for merge_idx,(merge_method, merge_min_id) in enumerate(final_merge_combos):
            
            #get labels
            clusterer = AgglomerativeClustering(metric='precomputed',
                                                linkage=merge_method,
                                                distance_threshold=1-merge_min_id,
                                                n_clusters=None)
            rep_seqs.loc[:,f'{id_col}_{merge_idx}'] = clusterer.fit_predict(dists)
        
        #merge on final cluster info
        vhh = vhh.merge(rep_seqs[[id_col]+[f'{id_col}_{merge_idx}' for merge_idx in range(len(final_merge_combos))]],on=id_col,how='left')

        #update within-cluster distances if needed for scoring
        # if args.score_fraction < 1:

        #     #get distance function params used in the clusterings that were merged
        #     dist_param_idx = int(id_col.split('_')[3])
        #     dist_params = cdr3_dist_combos[dist_param_idx]
            
        #     representatives = {} #use this to store representatives for large clusters
        #     saved_merge_dists = {} #use this to hold onto farthest within-cluster distance data from merges to re-use as needed
        #     for merge_idx,(merge_method, merge_min_id) in enumerate(final_merge_combos):
                
        #         #get all families that were merged
        #         merges = [group[id_col].tolist() for g,group in rep_seqs[[id_col,f'{id_col}_{merge_idx}']].groupby(f'{id_col}_{merge_idx}') if len(group)>1]
                
        #         #for each merge, calculate cross-family distances for merged families
        #         vhh[f'{dist_col}_{merge_idx}'] = np.nan
        #         for merge in merges:
        #             #iterate over all pairs of families
        #             for f1,fam1 in enumerate(merge[:-1]):
        #                 fam1_idx = vhh[id_col] == fam1
        #                 fam1_size = fam1_idx.sum()
        #                 fam1_cdr_list = [row[['CDR1','CDR2','CDR3']].tolist() for i,row in vhh[fam1_idx].iterrows()]
                        
        #                 for fam2 in merge[f1+1:]:
        #                     fam2_idx = vhh[id_col] == fam2
        #                     fam2_size = fam2_idx.sum()
                            
        #                     #calculate dists if not already calculated
        #                     if (fam1,fam2) not in saved_merge_dists:
                                
        #                         #if number of calculations is small, do them all
        #                         if (fam1_size<=args.min_cluster_size_shortcut) and (fam2_size<=args.min_cluster_size_shortcut):
        #                             fam2_cdr_list = [row[['CDR1','CDR2','CDR3']].tolist() for i,row in vhh[fam2_idx].iterrows()]
                                    
        #                             dists = np.array(pool.starmap(ANARCI_dist_row,((cdr,fam1_cdr_list,[1,1,dist_params[0]],[dist_params[1]]*3) for cdr in fam2_cdr_list),chunksize=10))
                                    
        #                             fam1_max = np.max(dists,axis=0)
        #                             fam2_max = np.max(dists,axis=1)
        #                         #otherwise, take a shortcut
        #                         else:
        #                             #if fam1 is too big, get representatives
        #                             if fam1_size > args.min_cluster_size_shortcut:
        #                                 #recycle cluster1 reps if already calculated
        #                                 if fam1 in representatives:
        #                                     reps1 = representatives[fam1]
        #                                 else:
        #                                     reps1 = get_mmseqs_reps(vhh.loc[fam1_idx,:],args.min_id_for_approx_dist,ncpus,tmp_dir)
        #                                     representatives[fam1] = reps1
        #                             #otherwise make each sequence its own rep
        #                             else:
        #                                 reps1 = pd.concat([vhh.loc[fam1_idx,'sequence_id']]*2,axis=1)
        #                                 reps1.columns = ['sequnce_id','mmseqs_id']
                                        
        #                             #if fam2 is too big, get representatives
        #                             if fam2_size > args.min_cluster_size_shortcut:
        #                                 if fam2 in representatives:
        #                                     reps2 = representatives[fam2]
        #                                 else:
        #                                     reps2 = get_mmseqs_reps(vhh.loc[fam2_idx,:],args.min_id_for_approx_dist,ncpus,tmp_dir)
        #                                     representatives[fam2] = reps2
        #                             #otherwise make each sequence its own rep
        #                             else:
        #                                 reps2 = pd.concat([vhh.loc[fam2_idx,'sequence_id']]*2,axis=1)
        #                                 reps2.columns = ['sequnce_id','mmseqs_id']
                                    
        #                             #get list of all representatives
        #                             reps1_names = reps1['mmseqs_id'].unique()
        #                             reps2_names = reps2['mmseqs_id'].unique()
                                    
        #                             #get CDR info from reps
        #                             vhh = vhh.set_index('sequence_id')
        #                             fam1_rep_cdrs = [row[['CDR1','CDR2','CDR3']].tolist() for i,row in vhh.loc[reps1_names,:].iterrows()]
        #                             fam2_rep_cdrs = [row[['CDR1','CDR2','CDR3']].tolist() for i,row in vhh.loc[reps2_names,:].iterrows()]
        #                             vhh = vhh.reset_index(drop=False)
                                    
        #                             #calculate dists to fam1 reps
        #                             dists = np.array(pool.starmap(ANARCI_dist_row,((cdr,fam2_rep_cdrs,[1,1,dist_params[0]],[dist_params[1]]*3) for cdr in fam1_rep_cdrs),chunksize=10))
        #                             fam1_max = np.max(dists,axis=1)
        #                             #assign dists to each member of the cluster they came from
        #                             fam1_max = pd.DataFrame({'mmseqs_id':reps1_names,'dist':fam1_max})
        #                             fam1_max = reps1.merge(fam1_max,on='mmseqs_id',how='left')['dist'].values

        #                             #calculate dists to fam2 reps
        #                             dists = np.array(pool.starmap(ANARCI_dist_row,((cdr,fam1_rep_cdrs,[1,1,dist_params[0]],[dist_params[1]]*3) for cdr in fam2_rep_cdrs),chunksize=10))
        #                             fam2_max = np.max(dists,axis=1)
        #                             #assign dists to each member of the cluster they came from
        #                             fam2_max = pd.DataFrame({'mmseqs_id':reps2_names,'dist':fam2_max})
        #                             fam2_max = reps2.merge(fam2_max,on='mmseqs_id',how='left')['dist'].values

        #                         #save farthest dists
        #                         saved_merge_dists[(fam1,fam2)] = fam1_max
        #                         saved_merge_dists[(fam2,fam1)] = fam2_max
        #                     #otherwise use saved dists
        #                     else:
        #                         fam1_max = saved_merge_dists[(fam1,fam2)]
        #                         fam2_max = saved_merge_dists[(fam2,fam1)]
                            
        #                     #find new farthest for each of the sequences in the merged families
        #                     vhh.loc[fam1_idx,f'{dist_col}_{merge_idx}'] = np.fmax(vhh.loc[fam1_idx,f'{dist_col}_{merge_idx}'],fam1_max)
        #                     vhh.loc[fam2_idx,f'{dist_col}_{merge_idx}'] = np.fmax(vhh.loc[fam2_idx,f'{dist_col}_{merge_idx}'],fam2_max)
                            
        #         # compare new farthest distances to old ones and fill in those that didn't need to be updated
        #         vhh[f'{dist_col}_{merge_idx}'] = vhh[[dist_col,f'{dist_col}_{merge_idx}']].max(axis=1)
                
        #delete original column and related furthest distance column since it is no longer needed
        del vhh[id_col]
        # del vhh[dist_col]
        
        #defragment by copying df (and reset index)
        vhh = vhh.copy()
    
    #reset index
    vhh = vhh.reset_index(drop=False)
    
    #save checkpoint
    vhh.to_csv(f'{tmp_dir}/intermediate.csv')
    
    print(f'Finished merges, starting quality filtering.', flush=True)
        
    #############
    ## filter individual clusterings
    #############
    
    id_cols = [col for col in vhh.columns if 'clone_id' in col]
    #skip filtering if all columns will be kept
    if args.score_fraction < 1:
        
        dist_param_map = {col:tuple(cdr3_dist_combos[int(col.split('_')[3])]) for col in id_cols}
        score_df = weighted_ANARCI_silhouette(vhh, id_cols, args.score_top_N, args.min_clust_reps, args.max_extra_reps, dist_param_map, args.sample_N_pairs, pool, ncpus, tmp_dir+'_mmseqs')

        score_df = pd.DataFrame(score_df,columns=['name','score']).sort_values(by='score',ascending=False)
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
    
def get_approx_furthest_dist(query_cdrs,target_df,top_N,weight_scheme,max_len_diffs):
    
    target_df = target_df.copy()
    target_df['selected'] = False
    
    #collect candidate distant seqs
    for cdr_idx in range(1,4):
        #calculate single-CDR dist from query to each of the targets
        cdrs = target_df[f'CDR{cdr_idx}'].unique().tolist()
        cdr_dists = []
        for cdr in cdrs:
            dist = distance(query_cdrs[cdr_idx-1],cdr)/max(len(cdr),len(query_cdrs[cdr_idx-1]))
            cdr_dists.append([cdr,dist])
        cdr_dists = pd.DataFrame(cdr_dists,columns=['CDR','dist'])
        
        #get top_N most different cdrs
        selected_cdrs = set(cdr_dists.sort_values(by='dist',ascending=False).head(top_N)['CDR'])
        #add any target with matching CDR to the set to check further
        target_df['selected'] = target_df['selected'] | target_df[f'CDR{cdr_idx}'].map(lambda x: x in selected_cdrs)
    
    #get selected sequences
    reduction_factor = target_df['selected'].mean()
    target_df = target_df[target_df['selected']]
    
    #calculate distances
    max_dist = np.max([weighted_anarci_dist(query_cdrs,row[['CDR1','CDR2','CDR3']].tolist(),weight_scheme,max_len_diffs) for i,row in target_df.iterrows()])
    
    return max_dist,reduction_factor

def get_mmseqs_reps(df,min_id,ncpus,tmp_dir):

    #run mmseqs to get cluster representatives
    os.mkdir(tmp_dir+'_mmseqs')
    
    with open(f'{tmp_dir}/mmseqs_in.fasta','w') as f:
        for i,row in df.iterrows():
            f.write(f">{row['sequence_id']}\n{''.join(row[['CDR1','CDR2','CDR3']])}\n") #Index here is the sequence_id
            
    #run mmseqs clustering
    result = subprocess.run(['singularity', 'exec', '/cluster/tufts/biocontainers/images/quay.io_biocontainers_mmseqs2:17.b804f--hd6d6fdc_1.sif', 'mmseqs', 'easy-cluster',
                            f'{tmp_dir}/mmseqs_in.fasta', f'{tmp_dir}_mmseqs/mmseqs_out', f'{tmp_dir}_mmseqs/tmp',
                             '-s', '7.5', #sensitivity set to max
                             '-c', str(max(0,(min_id - 0.2))), #set this lower than seq-id cutoff so it doesn't have an impact
                             '--min-seq-id', str(min_id), #only cluster seqs more similar than user input
                             '-e', 'inf', #max e-value to be clustered (set to keep all)
                             '-v', '1', #verbosity
                             '--cluster-steps', '3', #use faster cascaded clustering method
                             '--seq-id-mode', '0', #use alignment length to normalize
                             '--cluster-mode', '0', #use greedy setcover
                             '--alignment-mode', '3', #get percent identity, rather than alignment score
                             '--max-seqs', '300', #get top 1000 matches to each query
                             '--similarity-type', '2', #use sequence identity, not score
                             '--threads', str(ncpus)]) #how many cpus to use
    assert result.returncode == 0
    
    #read in clusters
    mmseqs_clusters = pd.read_csv(f'{tmp_dir}_mmseqs/mmseqs_out_cluster.tsv',names=['mmseqs_id','sequence_id'],sep='\t')
    
    #clean up
    subprocess.run(['rm','-r',tmp_dir+'_mmseqs'])
    
    #get reps
    rep_list = mmseqs_clusters["mmseqs_id"].unique()
    print(f'Reduced from {len(df)} sequences to {len(rep_list)} representatives.',flush=True)
    return mmseqs_clusters

if __name__ == '__main__':
    args = parser.parse_args()
    
    #set up for parallelization
    ncpus = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
    pool = Pool(processes=ncpus)
    print(f'Connected to pool with {ncpus} cpus.',flush=True)
    
    #parse output folder anf file names
    if args.out_file[0] != '/':
        args.out_file = './'+args.out_file
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
    
    #remove unnecessary column to save space
    del vhh['anarci_parse']
    
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